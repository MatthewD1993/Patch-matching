#include "ImageSequence.h"
#include <boost/concept_check.hpp>
#include "sintelGlobals.h"

//#define _Multiscale

class patchselect
{
public:
#ifdef _Multiscale
    static const int scales = 4;//hard!
#else
    static const int scales = 1;//hard!
#endif
    const int channels = 3;

    typedef cv::Mat_< cv::Vec<float,3> >  imgtype;

    class selectorMain
    {
    public:

        patchselect * ps;
        struct smPos
        {
            float x,y;
            int im;
        };
        smPos get ( )
        {
            smPos ret;
            ret.im = ( int ) randomIVal ( ps->_cntImages ); //start with image 1!
            ret.x = ( int ) randomIVal ( ps->_gt[ret.im].cols );
            ret.y = ( int ) randomIVal ( ps->_gt[ret.im].rows );
            return ret;
        }
    };

    class selectorSec
    {
    public:

        int cols,rows;
        int minDistq;
        struct ssPos
        {
            float x,y;
        };

        selectorSec ( int mindDist = 10 ) :minDistq ( mindDist*mindDist )
        {
        }

        virtual ssPos get ( const cv::Point2i & p, float scale )
        {
            ssPos ret;
            do
            {
                ret.x = randomIVal ( cols );
                ret.y = randomIVal ( rows );
            }
            while ( ( ret.x-p.x ) * ( ret.x-p.x ) + ( ret.y-p.y ) * ( ret.y-p.y ) < minDistq*scale*scale );

            return ret;
        }
    };

    class selectorClose : public selectorSec
    {
    public:

        int maxDistQ;
        int maxDist;

        selectorClose ( int mindDist = 10,int maxDist = 100 ) :selectorSec ( mindDist ), maxDist ( maxDist ), maxDistQ ( maxDist*maxDist )
        {

        }

        ssPos get ( const cv::Point2i & p, float scale )
        {
            ssPos ret;
            do
            {
                ret.x = randomIVal ( maxDist*scale*2+1 )- ( maxDist*scale ) + p.x;
                ret.y = randomIVal ( maxDist*scale*2+1 )- ( maxDist*scale ) +  p.y;

            }
            while ( ( ret.x-p.x ) * ( ret.x-p.x ) + ( ret.y-p.y ) * ( ret.y-p.y ) < minDistq*scale*scale || ret.x <0 || ret.y < 0 || ret.x >= cols || ret.y >= rows );

            return ret;
        }
    };


    int scaleCut[scales];

    typedef   std::pair < std::array<float, 6>, std:: array < imgtype, 2  > >  sampletype;
    typedef std::vector< sampletype  > samplelist;

    ImageSequence< imgtype >  _seq0;
    ImageSequence< imgtype >  _seq1;
    //std::vector < ImageSequence<cv::Mat1b> > _occ;
#ifdef _LEARN_DISP
    DisparitySequence _gt;
#else
    FlowSequence  _gt;
#endif

    int  _cntImages;
    int _patchsize;
    float _scale;

    samplelist _pos;//seq,img, x,y,x2,y2
    samplelist _neg;

    imgtype lpos,lneg,rpos,rneg;
    int _psreal;



    patchselect ( std::string  image1, std::string  image2,std::string flow, int cntImages, int patchsize = 32, float scale = 1, int offset=0 )
        : _cntImages ( cntImages ),_patchsize ( patchsize ),_scale ( scale )
    {

        _psreal = patchsize;
        for ( int i =1; i< scales; i++ ) _psreal*=2;
        //  _psreal*= scale;
        cout << "Size real:" << _psreal << endl;

        scaleCut[0]=0;
        for ( int i = 1, s = _psreal/4;  i< scales ; i++, s/=2 ) scaleCut[i]= scaleCut[i-1] + s;

        int cnt = 0;

        _seq0.offset = offset;
        _seq1.offset = offset;
        _seq0.setFileSequence ( image1 );
        _seq1.setFileSequence ( image2 );
        _gt.offset = offset;
        _gt._loadType = _gt.LOAD_TYPE_KITTI;
        _gt.setFileSequence ( flow );


        for ( int i =0; i< cntImages; i++ )
        {
            _gt ( i );
            //normalize Patches:
            for ( int k =0; k<2; k++ )
            {
                imgtype x =  k ? _seq0 ( i ) :_seq1 ( i );

                for ( int p=0; p<channels; p++ )
                {
                    double mean = 0;
                    for ( int k =0; k<x.rows; k++ )
                        for ( int l =0; l<x.cols; l++ ) mean += x ( k,l ) [p];

                    mean/= ( float ) ( x.rows*x.cols );

                    for ( int k =0; k<x.rows; k++ )
                        for ( int l =0; l<x.cols; l++ ) x ( k,l ) [p]-= mean;

                    double stdev= 0;
                    for ( int k =0; k<x.rows; k++ )
                        for ( int l =0; l<x.cols; l++ )   stdev += x ( k,l ) [p]*x ( k,l ) [p];

                    stdev= 1.f/sqrt ( stdev/ ( float ) ( x.rows*x.cols ) );

                    for ( int k =0; k<x.rows; k++ )
                        for ( int l =0; l<x.cols; l++ )
                        {
                            x ( k,l ) [p]*= stdev;
                            assert ( !isnan ( x ( k,l ) [p] ) );
                        }
                }
            }
            cv::copyMakeBorder ( _seq0[i], _seq0[i],_psreal/2, ( _psreal-1 ) /2,_psreal/2, ( _psreal-1 ) /2,cv::BORDER_REPLICATE );
            cv::copyMakeBorder ( _seq1[i], _seq1[i],_psreal/2, ( _psreal-1 ) /2,_psreal/2, ( _psreal-1 ) /2,cv::BORDER_REPLICATE );
        }
    }

    cv::Point2i getgtPos2p ( selectorMain::smPos & pos )
    {
        cv::Point2i res;

        res.x = pos.x + _gt ( pos.im ) ( pos.y,pos.x ) [0]+0.5f; //randomVal(2.f)-0.5f;//+0.5f;
        res.y = pos.y + _gt ( pos.im ) ( pos.y,pos.x ) [1]+0.5f; //randomVal(2.f)-0.5f;//+0.5f;
        if ( _gt[pos.im] ( pos.y,pos.x ) [0] > 1000000.f ) res.x =  res.y = -1000;
        return res;
    }


    void createTorchPtrDist ( float * ptr )
    {
        assert ( _pos.size() == _neg.size() );

        for ( int i =0; i<_pos.size(); i++ )
        {
            *ptr++ = _pos[i].first[0];
            *ptr++ = _neg[i].first[0];
        }
    }

    void createTorchPtrInfo ( float * ptr )
    {
        assert ( _pos.size() == _neg.size() );

        for ( int i =0; i<_pos.size(); i++ )
        {
            *ptr++ = _pos[i].first[2];
            *ptr++ = _pos[i].first[3];
            *ptr++ = _pos[i].first[4];
            *ptr++ = _pos[i].first[5];
            *ptr++ = _pos[i].first[4]-_pos[i].first[2];
            *ptr++ = _pos[i].first[5]-_pos[i].first[3];

            *ptr++ = _neg[i].first[2];
            *ptr++ = _neg[i].first[3];
            *ptr++ = _neg[i].first[4];
            *ptr++ = _neg[i].first[5];
            *ptr++ = _neg[i].first[4]-_neg[i].first[2];
            *ptr++ = _neg[i].first[5]-_neg[i].first[3];
        }
    }



    void createTorchPtr ( float * ptr ) // pos == 2 -> pos + neg
    {
        float *pt[3];
        pt[0]= ptr;
        pt[1]= ptr+ ( 1*_patchsize*_patchsize );
        pt[2]= ptr+ ( 2*_patchsize*_patchsize );

        assert ( _pos.size() == _neg.size() );

        samplelist * arr;
        arr =&_pos;
        else arr = &_neg;

        for ( int i =0; i<arr->size(); i++ )
        {
            for ( int ss =0; ss<2; ss++ )
            {
                imgtype x = ( *arr ) [i].second[ss];
                assert ( x.rows == _patchsize );

                for ( int j= 0; j< x.rows; j++ )
                    for ( int k= 0; k< x.cols; k++ )
                    {
                        *pt[0]++ = ( float ) x ( j,k ) [0];
                        *pt[1]++ = ( float ) x ( j,k ) [1];
                        *pt[2]++ = ( float ) x ( j,k ) [2];
                    }
                pt[0]+= ( _patchsize*_patchsize*2 );
                pt[1]+= ( _patchsize*_patchsize*2 );
                pt[2]+= ( _patchsize*_patchsize*2 );

            }
           
            if ( arr == &_pos )
            {
                arr = &_neg;
                i--;
            }
            else  arr = &_pos;
            
        }
        assert ( pt[0] == ptr + ( _patchsize*_patchsize*2*channels*2*arr->size() ) );
    }

    void reset()
    {
        _pos.resize ( 0 );
        _neg.resize ( 0 );
    }

    void add ( int cnt,  std::vector< selectorMain*> sm,  std::vector< selectorSec*> ss, bool addPositive = true, bool _50p_difficult = true )
    {
        int s =  0;//_scale;//indexx[sfac];
        assert ( _scale ==1 );

        for ( int i =0; i<sm.size(); i++ ) sm[i]->ps = this;

        for ( int i =0; i< cnt; i++ )
        {
            auto pos1 = sm[  randomIVal ( sm.size() ) ]->get();
            cv::Point2i pos2p = getgtPos2p ( pos1 );

            if ( addPositive && ( pos2p.x < 0 || pos2p.y < 0 || pos2p.x >= _gt[pos1.im].cols || pos2p.y >= _gt[pos1.im].rows /*||  _occ[pos1.seq][pos1.im] ( pos1.y,pos1.x )*/ ) )
            {
                i--;
                continue;
            }
            assert ( pos2p.y < _gt[pos1.im].rows );

            float gx = _gt ( pos1.im ) ( pos1.y,pos1.x ) [0];
            float gy = _gt ( pos1.im ) ( pos1.y,pos1.x ) [0];

            float gs =  sqrtf ( gx*gx+gy*gy );

#ifdef	_Multiscale
            int indexx[] = {0,0,0,1,1,1,2,2,2,2,2};        // int indexx[] = {0,0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3};
            assert ( scales ==4 );
            int sfac = randomIVal ( 11 );
            s = indexx[sfac];
#endif

            if ( addPositive )
            {
                sampletype  p1;
                p1.first = {0,pos1.im,pos1.x, pos1.y,pos2p.x,pos2p.y};

                imgtype img1 = _seq0[pos1.im] ( cv::Rect ( pos1.x,pos1.y,_psreal,_psreal ) );
                imgtype img2 = _seq1[pos1.im] ( cv::Rect ( pos2p.x,pos2p.y,_psreal,_psreal ) );

                cv::resize ( img1 ( cv::Rect ( scaleCut[s], scaleCut[s], img1.cols-scaleCut[s]*2, img1.rows-scaleCut[s]*2 ) ),p1.second[0] ,cv::Size ( _patchsize,_patchsize ),0,0,CV_INTER_AREA );
                cv::resize ( img2 ( cv::Rect ( scaleCut[s], scaleCut[s], img1.cols-scaleCut[s]*2, img1.rows-scaleCut[s]*2 ) ),p1.second[1] ,cv::Size ( _patchsize,_patchsize ),0,0,CV_INTER_AREA );
                //cv::putText(p1.second[0], std::to_string(s), cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                _pos.push_back ( p1 );
                assert ( pos1.x >=0 || pos1.y >= 0 && pos2p.x >= 0 && pos2p.y >=0 && ( pos2p.x-pos1.x- _gt[pos1.im] ( pos1.y,pos1.x ) [0] ) <1.1f );
            }

            if ( ss.size() )   //TODO: set size
            {
                int choosen =  randomIVal ( ss.size() );
                ss[choosen]->cols = _gt[pos1.im].cols;
                ss[choosen]->rows = _gt[pos1.im].rows;
                
#ifdef _Multiscale
		const int scaledx []= {8,4,2,1};
                auto pos2n = ss[choosen]->get ( pos2p,scaledx[s] );
#else
                assert ( _scale==1 );
                auto pos2n = ss[choosen]->get ( pos2p, _scale );
#endif

                float disttopos = sqrtf ( ( pos2n.x - pos2p.x ) * ( pos2n.x - pos2p.x ) + ( pos2n.y - pos2p.y ) * ( pos2n.y - pos2p.y ) );
                sampletype  p1;
                p1.first = {disttopos,pos1.im,pos1.x, pos1.y,pos2n.x,pos2n.y};

                imgtype  img1  = _seq0[pos1.im] ( cv::Rect ( pos1.x,pos1.y,_psreal,_psreal ) );
                imgtype  img2  = _seq1[pos1.im] ( cv::Rect ( pos2n.x,pos2n.y,_psreal,_psreal ) );

                cv::resize ( img1 ( cv::Rect ( scaleCut[s], scaleCut[s], img1.cols-scaleCut[s]*2, img1.rows-scaleCut[s]*2 ) ),p1.second[0],cv::Size ( _patchsize,_patchsize ),0,0,CV_INTER_AREA );
                cv::resize ( img2 ( cv::Rect ( scaleCut[s], scaleCut[s], img1.cols-scaleCut[s]*2, img1.rows-scaleCut[s]*2 ) ),p1.second[1],cv::Size ( _patchsize,_patchsize ),0,0,CV_INTER_AREA );

                _neg.push_back ( p1 );
                assert ( pos1.x >=0 || pos1.y >= 0 && pos2n.x >= 0 && pos2n.y >=0 && ( pos2n.x-pos1.x- _gt[pos1.im] ( pos1.y,pos1.x ) [0] ) >2.1f );

            }


            if ( _50p_difficult && i%3 )
            {
                assert ( 0 );
                /*   assert ( addPositive );
                   cv::Mat1f res,res2;
                   assert ( _pos.size() == _neg.size() );

                   cv::matchTemplate ( _pos[_pos.size()-1].second[scales-1],_pos[_pos.size()-1].second[scales-1+scales],res,CV_TM_CCOEFF_NORMED );
                   cv::matchTemplate ( _neg[_pos.size()-1].second[scales-1],_neg[_pos.size()-1].second[scales-1+scales],res2,CV_TM_CCOEFF_NORMED );
                   if ( res2 ( 0,0 ) < res ( 0,0 ) )
                   {
                       _neg.pop_back();
                       _pos.pop_back();
                       i--;
                   }*/
            }
        }
    }


    void createDbgImgages ( int xSize )
    {

        assert ( _pos.size() %xSize ==0 );
        assert ( _neg.size() %xSize ==0 );

        int ypSize =  _pos.size() /xSize;
        int ynSize =  _neg.size() /xSize;

        lpos = imgtype ( ypSize*_patchsize, xSize*_patchsize );
        rpos = imgtype ( ypSize*_patchsize, xSize*_patchsize );

        lneg = imgtype ( ynSize*_patchsize, xSize*_patchsize );
        rneg = imgtype ( ynSize*_patchsize, xSize*_patchsize );

        int cnt = 0;
        for ( int i =0; i < ypSize ; i ++ )
        {
            for ( int j =0 ; j< xSize; j++ )
            {
                //    1         2        3     4      5
                // {0,pos1.im,pos1.x, pos1.y,pos2p.x,pos2p.y};
                _pos[cnt].second[0].copyTo ( lpos ( cv::Rect ( j*_patchsize, i*_patchsize, _patchsize, _patchsize ) ) );
                _pos[cnt].second[1].copyTo ( rpos ( cv::Rect ( j*_patchsize, i*_patchsize, _patchsize, _patchsize ) ) );

                cnt++;
            }
        }
        cnt = 0;
        for ( int i =0; i < ynSize ; i ++ )
        {
            for ( int j =0 ; j< xSize; j++ )
            {
                _neg[cnt].second[0].copyTo ( lneg ( cv::Rect ( j*_patchsize, i*_patchsize, _patchsize, _patchsize ) ) );
                _neg[cnt].second[1].copyTo ( rneg ( cv::Rect ( j*_patchsize, i*_patchsize, _patchsize, _patchsize ) ) );
                cnt++;
            }
        }
    }


};

/*
int patchselecttest()
{

    int usedSeq=2;
      patchselect ps ( {sintelFiles ( 0, usedSeq )}, {sintelFlowFiles ( usedSeq )}, {sintelOccFiles ( usedSeq )},{seqLengths[usedSeq]},32,usedSeq );
      //patchselect::selectorMain()
      std::vector<patchselect::selectorMain*>  sml;
      sml.push_back ( new patchselect::selectorMain() );

      std::vector<patchselect::selectorSec*>  ssl;
      ssl.push_back ( new patchselect::selectorSec ( 8 ) );
      ssl.push_back ( new patchselect::selectorClose ( 8,100 ) );
      ssl.push_back ( new patchselect::selectorClose ( 8,20 ) );

      ps.add ( 100*100, sml,ssl );
      ps.createImgages ( 100 );

      cv::imshow ( "lpos",ps.lpos );
      cv::imshow ( "rpos",ps.rpos );

      cv::imshow ( "lneg",ps.lneg );
      cv::imshow ( "rneg",ps.rneg );

      cv::waitKey ( 0 );

}*/

