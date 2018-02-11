#pragma once
#include "basicdefinitions.h"
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <zlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>

//#include <string>
//#include <iostream>

#define SAVE_FILESEQUENCE_FAST_OPERATOR

template <typename t>
class FileSequence
{
    virtual void  saveFile_ ( std::string filename,const t & img ) =0;
    /** Load File if available, otherwise return valid object*/
    virtual bool loadFile ( std::string filename, t & obj ) =0;
    virtual bool isLoadable_ ( std::string filename )
    {
        TODO_STOP; //TODO: Test existence of file
    }

protected:

    std::string _ending;
    std::vector< t > _data;
    std::vector< char > _state;
    int _seqZeros;
    std::string _sequence1, _sequence2;

public:

    int offset;
    FileSequence():offset(0) {}


    FileSequence ( std::string sequence ):offset(0)
    {
        setFileSequence ( sequence );
    }

    void setFileSequence ( std::string sequence )
    {
        _data.clear();
        int p = -1;
        for ( int i =0; i<sequence.size(); i++ ) {if ( sequence[i]=='%' ) p = i;};
        assert ( p>=0 );
        _seqZeros = sequence[p+1]-'0';
        _sequence1 = sequence.substr ( 0,p );
        _sequence2 = sequence.substr ( p+2 );
        p = -1;
        for ( int i =0; i<sequence.size(); i++ ) if ( sequence[i]=='.' ) p = i;
        if ( p>=0 ) _ending = sequence.substr ( p+1 );
    }

    std::string filename ( int at )
    {
        at += offset;
        std::stringstream filename;
        std::stringstream ss_zero_padded_num;
        ss_zero_padded_num << std::setw ( _seqZeros ) << std::setfill ( '0' ) << at;
        return _sequence1 + ss_zero_padded_num.str() + _sequence2;
    }

    t & loadFile ( int at )
    {
        bool success = loadFile ( filename(at), _data[at] );
        if ( !success ) cout <<"Warning: " << filename ( at ) << " does not exist."<<endl;
        else _state[at] |=2;
        _state[at] |=1;
        return _data[at];
    }

    bool isLoadable ( int at )
    {
        return isLoadable_ ( filename ( at ) );
    }


    void saveFile ( int at, t img )
    {
        // if ( _data.size() <= at ) _data.resize ( at+1 );
        // _data[at] = img;
        saveFile_ ( filename ( at ), img );
    }

    void saveFile ( int at )
    {
        saveFile_ ( filename ( at ), _data[at] );
    }

    void saveAll()
    {
        for ( int i =0; i<_data.size(); i++ ) if ( _state[i] & 1 ) saveFile ( i );
    }

    int size() { return _data.size(); }

    /**  Fast Access  */
    t & operator [] ( int i )
    {
    #ifdef SAVE_FILESEQUENCE_FAST_OPERATOR
            if ( _data.size() <= i ) resize ( i+1 );
            if ( ! ( _state[i] & 1 ) )  _data[i] = loadFile ( i );
    #endif
        return _data[i];
    }

    /**  Save Access  */
    t & operator () ( int i )
    {
        if ( _data.size() <= i ) resize ( i+1 );
        if ( ! ( _state[i] & 1 ) )  _data[i] = loadFile ( i );
        return _data[i];
    }

    void resize ( int size )
    {
        _data.resize ( size );
        _state.resize ( size,0 );
    }

    void resizeNoLoading ( int size )
    {
        _data.resize ( size );
        _state.resize ( size,1 );
    }

};


template <typename t>
class ImageSequence : public FileSequence<t>
{
    bool _lab;

    virtual void  saveFile_ ( std::string filename,const t & img )
    {
        cv::imwrite ( filename, img );
    }

    bool loadFile ( std::string filename, t & obj )
    {
        if ( _lab &&  t().channels() !=1 )
        {
            assert ( t().channels() == 3 );
            obj =  cv::imread ( filename,1 );
            if ( obj.empty() ) return false;
            if ( obj.depth() == CV_32F ) {obj/=255.f;}
            cv::cvtColor ( obj,obj, CV_BGR2Lab ); //Chengbiao: change RGB to BGR
            return ( !obj.empty() );
        }

        if ( t().channels() ==1 ) obj =  cv::imread ( filename, 0 );
        else obj = cv::imread ( filename, 1 );
        if ( obj.depth() == CV_32F ) obj/=255.f;
        return ( !obj.empty() );
    }

public:
    ImageSequence (bool lab = true) : _lab ( lab ) { cout<<"Use Lab format image: "<< _lab << endl; }

    ImageSequence ( std::string sequence, bool lab = false ) : _lab ( lab )
    {
        this->setFileSequence ( sequence );
    }
};



template <typename t = cv::Mat>
class cvMatSequence: public  FileSequence<t>
{
public:

    cvMatSequence ( std::string sequence )
    {
        this->setFileSequence ( sequence );
    }
    bool loadFile ( std::string filename, t & obj )
    {
        std::ifstream file;
        file.open ( filename,  std::ios::in |  std::ios::binary );
        if ( !file.is_open() ) return false;

        short ident;
        uchar version,unused;
        int w, h, cvType;
        unsigned short channels, bytesByElement;

        file.read ( ( char* ) &ident,sizeof ( short ) );
        file.read ( ( char* ) &version,sizeof ( uchar ) );
        file.read ( ( char* ) &unused,sizeof ( uchar ) );
        assert ( ident == 24270 && version == 0 );

        file.read ( ( char* ) &w,sizeof ( int ) );
        file.read ( ( char* ) &h,sizeof ( int ) );
        file.read ( ( char* ) &cvType,sizeof ( int ) );
        file.read ( ( char* ) &channels,sizeof ( unsigned short ) );
        file.read ( ( char* ) &bytesByElement,sizeof ( unsigned short ) );

        obj = cv::Mat ( h,w,cvType );
        assert ( obj.channels() == channels );
        assert ( obj.elemSize() == bytesByElement );
        file.read ( ( char* ) obj.data,  bytesByElement*w*h );
        file.close();
        return true;
    }

    void  saveFile_ ( std::string filename, const t & obj )
    {
        assert ( obj.isContinuous() );
        std::ofstream file;
        file.open ( filename,  std::ios::out |  std::ios::binary );
        assert ( file.is_open() );

        short ident = 24270;
        uchar version=0,unused=0;
        unsigned short channels= obj.channels(), bytesByElement = obj.elemSize();

        file.write ( ( char* ) &ident,sizeof ( short ) );
        file.write ( ( char* ) &version,sizeof ( uchar ) );
        file.write ( ( char* ) &unused,sizeof ( uchar ) );

        file.write ( ( char* ) & ( obj.cols ),sizeof ( int ) );
        file.write ( ( char* ) & ( obj.rows ),sizeof ( int ) );
        int type = obj.type();
        file.write ( ( char* ) & ( type ),sizeof ( int ) );
        file.write ( ( char* ) &channels,sizeof ( unsigned short ) );
        file.write ( ( char* ) &bytesByElement,sizeof ( unsigned short ) );

        file.write ( ( char* ) obj.data,  bytesByElement*obj.cols*obj.rows );
        file.close();
    }
};


class FlowSequence:  public FileSequence<cv::Mat2f>
{
public:
    enum loadType
    {
        LOAD_TYPE_UNDEFINED,
        LOAD_TYPE_FROM_FILENAME,
        LOAD_TYPE_MPI,
        LOAD_TYPE_KITTI
    };

    loadType _loadType;

    FlowSequence() : _loadType ( LOAD_TYPE_UNDEFINED ){}

    FlowSequence ( std::string sequence, loadType lType = LOAD_TYPE_FROM_FILENAME ) :  _loadType ( lType )
    {
        setFileSequence ( sequence );
        if ( _ending=="flo" && lType == LOAD_TYPE_FROM_FILENAME ) _loadType = LOAD_TYPE_MPI;
	    if ( _ending=="png" && lType == LOAD_TYPE_FROM_FILENAME ) _loadType = LOAD_TYPE_KITTI;
        assert ( _loadType != LOAD_TYPE_FROM_FILENAME && _loadType != LOAD_TYPE_UNDEFINED );
    }


private:
    void  saveFile_ ( std::string filename, const cv::Mat2f & img )
    {
        std::ofstream file;
        file.open ( filename,  std::ios::out |  std::ios::binary );
        assert ( file.is_open() );

        if ( _loadType == LOAD_TYPE_MPI )
        {
            assert ( _ending == "flo" );
            float x=202021.25;
            int w = img.cols;
            int h = img.rows;

            file.write ( ( char* ) &x,sizeof ( float ) );
            file.write ( ( char* ) &w,sizeof ( int ) );
            file.write ( ( char* ) &h,sizeof ( int ) );

            for ( int i =0; i<h; i++ ) for ( int j =0; j<w; j++ )
                {
                    file.write ( ( char* ) &img ( i,j ) [0],sizeof ( float ) );
                    file.write ( ( char* ) &img ( i,j ) [1],sizeof ( float ) );
                }
        }
        else assert ( 0 );
    }

    bool loadFile ( std::string filename, cv::Mat2f & res )
    {
        std::ifstream file;
        file.open ( filename,  std::ios::in |  std::ios::binary );
        if ( !file.is_open() ) return false;

        if ( _loadType == LOAD_TYPE_MPI )
        {
            assert ( _ending == "flo" );
            float x;
            int w,h;

            file.read ( ( char* ) &x, sizeof( float ) );
            file.read ( ( char* ) &w, sizeof( int ) );
            file.read ( ( char* ) &h, sizeof( int ) );

            assert ( x == 202021.25 );
            res =   cv::Mat2f ( h,w );
            for ( int i =0; i<h; i++ ) for ( int j =0; j<w; j++ )
                {
                    file.read ( ( char* ) &res ( i,j )[0], sizeof( float ) );
                    file.read ( ( char* ) &res ( i,j )[1], sizeof( float ) );
                }
        }


        else if ( _loadType == LOAD_TYPE_KITTI )
        {
	    assert ( _ending == "png" );
            cv::Mat3f obj = cv::imread ( filename,CV_LOAD_IMAGE_UNCHANGED );
            cv::Mat3f  x = ( obj -32768 ) /64.f;

            std::vector<cv::Mat1f> y, y2;
            cv::split ( x,y );
            y2.push_back ( y[2] );
            y2.push_back ( y[1] );
            cv::merge ( y2,res );
            forcvmat ( i,j,res )
            {
                if ( x ( i,j ) [0] ==-512 )  res( i,j )[0] = res( i,j )[1] = std::numeric_limits<float>::infinity();
            }
        }
        else
        {
            std::cerr << "Unsupported type. Ending is:" << _ending << endl;
            assert ( 0 );
        }
        return true;
    }

};



