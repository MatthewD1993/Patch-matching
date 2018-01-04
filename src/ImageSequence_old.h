#pragma once
#include "basicdefinitions.h"
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <zlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.h>

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
    FileSequence():offset(0)
    {
	
    }

    FileSequence ( std::string sequence ):offset(0)
    {
        setFileSequence ( sequence );
    }

    void setFileSequence ( std::string sequence )
    {
        _data.clear();
        int p = -1;
        for ( int i =0; i<sequence.size(); i++ ) if ( sequence[i]=='%' ) p = i;
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
        bool success = loadFile ( filename ( at ),_data[at] );
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

    int size()
    {
        return _data.size();
    }
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
            // assert(0);
            assert ( t().channels() ==3 );
	    cout << filename << endl;
            obj =  cv::imread ( filename,1 );
            if ( obj.empty() ) return false;
            if ( obj.depth() == CV_32F ) obj/=255.f;
            cv::cvtColor ( obj,obj,CV_RGB2Lab );
            return ( !obj.empty() );
        }
        if ( t().channels() ==1 ) obj =  cv::imread ( filename,0 );
        else obj = cv::imread ( filename,1 );
        if ( obj.depth() == CV_32F ) obj/=255.f;
        return ( !obj.empty() );
    }
public:
    ImageSequence() : _lab ( true )
    {
    }

    ImageSequence ( std::string sequence, bool lab = false ) : _lab ( lab )
    {
        this->setFileSequence ( sequence );
    }
};



template <typename t = cv::Mat>
class cvMatSequene: public  FileSequence<t>
{
public:

    cvMatSequene ( std::string sequence )
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


/*
template <int size>
class FlowerrormapsSequence:  public FileSequence< flowerrormaps<size> >
{
public:

    enum storageType : unsigned int
    {
        STD=0b00,
        REDUCED=0b01,
        //10,11 -> For later use
        NOERR = 0b100,
        COMPRESSED = 0b1000
                     // xx000 -> for compression levels
    };
    unsigned int _type;
    int _flag;



    FlowerrormapsSequence ( std::string sequence, storageType type = COMPRESSED, int flag = 0 ) : _type ( type ),_flag ( flag )
    {
        this->setFileSequence ( sequence );
    }

private:

    void  saveFile_ ( std::string filename,  const flowerrormaps<size> & img )
    {
        std::ofstream file;
        gzFile cfile;

        float x = 451878.25;
        int w = img.flows[0].cols;
        int h = img.flows[0].rows;

        int tsize = size;

        if ( _type & COMPRESSED )
        {
            cfile = gzopen ( filename.c_str(),"wb" );
	    if(cfile ==NULL) assert(0);
            gzwrite ( cfile, ( void* ) &x,sizeof ( float ) );
            gzwrite ( cfile, ( void* ) &_type,sizeof ( int ) );
            gzwrite ( cfile, ( void* ) &_flag,sizeof ( int ) );
            gzwrite ( cfile, ( void* ) &tsize,sizeof ( int ) );
            gzwrite ( cfile, ( void* ) &w,sizeof ( int ) );
            gzwrite ( cfile, ( void* ) &h,sizeof ( int ) );
        }
        else
        {
            file.open ( filename,  std::ios::out |  std::ios::binary );
            assert ( file.is_open() );
            file.write ( ( char* ) &x,sizeof ( float ) );
            file.write ( ( char* ) &_type,sizeof ( int ) );
            file.write ( ( char* ) &_flag,sizeof ( int ) );
            file.write ( ( char* ) &tsize,sizeof ( int ) );
            file.write ( ( char* ) &w,sizeof ( int ) );
            file.write ( ( char* ) &h,sizeof ( int ) );
        }

        cv::Mat2s shortflow;

        for ( int i =0; i< size; i++ )
        {
            assert ( img.flows[i].isContinuous() );

            if ( _type  & COMPRESSED )
            {
	       if ( _type  & REDUCED )
	       {
                img.flows[i].convertTo ( shortflow, CV_16SC2 );
                assert ( shortflow.isContinuous() );
                gzwrite ( cfile, ( void* ) shortflow.data,sizeof ( short ) *2*w*h );
	       }
	       else  gzwrite ( cfile, ( void* ) img.flows[i].data,sizeof ( float ) *2*w*h );
            }
            else
            {
                if ( _type  & REDUCED )
                {
                    img.flows[i].convertTo ( shortflow, CV_16SC2 );
                    assert ( shortflow.isContinuous() );
                    file.write ( ( char* ) shortflow.data, sizeof ( short ) *2*w*h ) ;
                }
                else file.write ( ( char* ) img.flows[i].data, sizeof ( float ) *2*w*h ) ;
            }

        }

        int closeval = 1234567905;
        if ( _type & COMPRESSED )
        {
            gzwrite ( cfile, ( void* ) &closeval,sizeof ( int ) );
            gzclose ( cfile );
        }
        else
        {
            file.write ( ( char* ) &closeval,sizeof ( int ) );
            file.close();
        }

        if ( ! ( _type& NOERR ) )
        {
            if ( _type  & COMPRESSED )
            {
                gzFile cfile;
                cfile = gzopen ( ( filename+"e" ).c_str(),"wb" );
                for ( int i =0; i< size; i++ )   gzwrite ( cfile, ( char* ) img.errors[i].data, sizeof ( float ) *w*h ) ;
                gzwrite ( cfile, ( void* ) &closeval,sizeof ( int ) );
                gzclose ( cfile );
            }
            else
            {
                std::ofstream file;
                file.open ( filename+"e",  std::ios::out |  std::ios::binary );
                for ( int i =0; i< size; i++ )   file.write ( ( char* ) img.errors[i].data, sizeof ( float ) *w*h );
                file.write ( ( char* ) &closeval,sizeof ( int ) );
                file.close();
            }

        }

    }

    bool isLoadable_ ( std::string filename )
    {
        std::ifstream file;
        gzFile cfile;

        float x;
        int w,h;
        int t,f;
        int tsize=-1;
        if ( _type & COMPRESSED )
        {
            cfile = gzopen ( filename.c_str(),"rb" );
	    if(cfile ==NULL) return false;
            gzread ( cfile, ( void* ) &x,sizeof ( float ) );
            gzread ( cfile, ( void* ) &t,sizeof ( int ) );
            gzread ( cfile, ( void* ) &f,sizeof ( int ) );
            gzread ( cfile, ( void* ) &tsize,sizeof ( int ) );
            gzread ( cfile, ( void* ) &w,sizeof ( int ) );
            gzread ( cfile, ( void* ) &h,sizeof ( int ) );
        }
        else
        {
            file.open ( filename,  std::ios::in |  std::ios::binary );
            if ( !file.is_open() ) return false;
            file.read ( ( char* ) &x,sizeof ( float ) );
            file.read ( ( char* ) &t,sizeof ( int ) );
            file.read ( ( char* ) &f,sizeof ( int ) );
            file.read ( ( char* ) &tsize,sizeof ( int ) );
            file.read ( ( char* ) &w,sizeof ( int ) );
            file.read ( ( char* ) &h,sizeof ( int ) );
        }
        if ( _type & COMPRESSED ) gzclose ( cfile );
        else file.close();

        if ( tsize<size ) return false;
        if ( x != 451878.25 ) return false;
        if ( t != _type ) return false;
        if ( f != _flag )
        {
            cout << "Warning:"+filename+" is loadable but user set flag is wrong"<<endl;
            return false;
        }
        return true;

    }


    bool  loadFile ( std::string filename,  flowerrormaps<size> & obj )
    {
        cv::Mat2s shortflow;

        std::ifstream file;
        gzFile cfile;

        float x;
        int w,h;
        int t,f;
        int tsize=-10;
        if ( _type & COMPRESSED )
        {
            cfile = gzopen ( filename.c_str(),"rb" );
	    if(cfile ==NULL)
	    {
	      cout << filename << " not available"<<endl;
	      exit(1);
	    }
            gzread ( cfile, ( void* ) &x,sizeof ( float ) );
            gzread ( cfile, ( void* ) &t,sizeof ( int ) );
            gzread ( cfile, ( void* ) &f,sizeof ( int ) );
            gzread ( cfile, ( void* ) &tsize,sizeof ( int ) );
            gzread ( cfile, ( void* ) &w,sizeof ( int ) );
            gzread ( cfile, ( void* ) &h,sizeof ( int ) );
        }
        else
        {
            file.open ( filename,  std::ios::in |  std::ios::binary );
            if ( !file.is_open() ) return false;
            file.read ( ( char* ) &x,sizeof ( float ) );
            file.read ( ( char* ) &t,sizeof ( int ) );
            file.read ( ( char* ) &f,sizeof ( int ) );
            file.read ( ( char* ) &tsize,sizeof ( int ) );
            file.read ( ( char* ) &w,sizeof ( int ) );
            file.read ( ( char* ) &h,sizeof ( int ) );
        }
        assert ( x == 451878.25 );
        assert ( t == _type );
	if(tsize<size )_iout2(tsize,size);
	assert ( tsize>=size );

        shortflow = cv::Mat2s ( h,w );
        assert ( shortflow.isContinuous() );

        for ( int i =0; i< size; i++ )
        {


            if ( _type & COMPRESSED )
            {
	       if ( _type & REDUCED )
	       {
                gzread ( cfile, ( void* ) shortflow.data,sizeof ( short ) *2*w*h );
                shortflow.convertTo ( obj.flows[i],CV_32FC2 );
	       }
	       else
	       {
		  obj.flows[i] =  cv::Mat2f ( h,w );
                  assert ( obj.flows[i].isContinuous() );
		  gzread ( cfile, ( void* ) obj.flows[i].data,sizeof ( float) *2*w*h );
	       }
            }
            else
            {
                if ( _type & REDUCED )
                {
                    file.read ( ( char* ) shortflow.data,sizeof ( short ) *2*w*h );
                    shortflow.convertTo ( obj.flows[i],CV_32FC2 );
                }
                else
                {
                    obj.flows[i] =  cv::Mat2f ( h,w );
                    assert ( obj.flows[i].isContinuous() );
                    file.read ( ( char* ) obj.flows[i].data,sizeof ( float ) *2*w*h );

                }
            }
        }
        if ( _type & COMPRESSED ) gzclose ( cfile );
        else file.close();

        if ( ! ( _type & NOERR ) )
        {
            if ( _type  & COMPRESSED )
            {
                gzFile cfile;
                cfile = gzopen ( ( filename+"e" ).c_str(),"rb" );
                for ( int i =0; i< size; i++ )
                {
                    obj.errors[i] =  cv::Mat1f ( h,w );
                    assert ( obj.errors[i].isContinuous() );
                    gzread ( cfile, ( void* ) obj.errors[i].data,sizeof ( float ) *w*h );
                }
                gzclose ( cfile );
            }
            else
            {
                std::ifstream file;
                file.open ( filename+"e",  std::ios::in |  std::ios::binary );

                for ( int i =0; i< size; i++ )
                {
                    obj.errors[i] =  cv::Mat1f ( h,w );
                    assert ( obj.errors[i].isContinuous() );
                    file.read ( ( char* ) obj.errors[i].data,sizeof ( float ) *w*h );
                }
                file.close();
            }

        }
        if ( f != _flag && _flag != 0 ) cout << "Warning:"+filename+" was loaded but user set flag is wrong"<<endl;
        return true;
    }


};*/

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

    FlowSequence() : _loadType ( LOAD_TYPE_UNDEFINED )
    {}

    FlowSequence ( std::string sequence, loadType lType = LOAD_TYPE_FROM_FILENAME ) :  _loadType ( lType )
    {
        setFileSequence ( sequence );
        if ( _ending=="flo" && lType == LOAD_TYPE_FROM_FILENAME ) _loadType = LOAD_TYPE_MPI;
	if ( _ending=="png" && lType == LOAD_TYPE_FROM_FILENAME ) _loadType = LOAD_TYPE_KITTI;
        assert ( _loadType != LOAD_TYPE_FROM_FILENAME && _loadType != LOAD_TYPE_UNDEFINED );
    }

    loadType _loadType;
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

            file.read ( ( char* ) &x,sizeof ( float ) );
            file.read ( ( char* ) &w,sizeof ( int ) );
            file.read ( ( char* ) &h,sizeof ( int ) );

            assert ( x == 202021.25 );
            res =   cv::Mat2f ( h,w );
            for ( int i =0; i<h; i++ ) for ( int j =0; j<w; j++ )
                {
                    file.read ( ( char* ) &res ( i,j ) [0],sizeof ( float ) );
                    file.read ( ( char* ) &res ( i,j ) [1],sizeof ( float ) );
                }
        }
        else if ( _loadType == LOAD_TYPE_KITTI )
        {
	   assert ( _ending == "png" );
            cv::Mat3f obj = cv::imread ( filename,CV_LOAD_IMAGE_UNCHANGED );
            cv::Mat3f  x = ( obj -32768 ) /64.f;

            std::vector<cv::Mat1f> y;
            cv::split ( x,y );
            std::vector<cv::Mat1f> y2;
            y2.push_back ( y[2] );
            y2.push_back ( y[1] );
            cv::merge ( y2,res );
            forcvmat ( i,j,res )
            {
                if ( x ( i,j ) [0] ==-512 )  res ( i,j ) [0]=res ( i,j ) [1]=std::numeric_limits< float >::infinity();
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



