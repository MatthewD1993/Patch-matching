#include "ImageSequence.h"
#include <boost/concept_check.hpp>
// #include "sintelGlobals.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
using namespace cv;

class patchselect
{
public:

    static const int scales = 1;//hard!
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

        selectorSec ( int mindDist = 10 ) :minDistq ( mindDist*mindDist ) {}

        virtual ssPos get ( const cv::Point2i & p, int scale )
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

        selectorClose ( int mindDist = 10,int maxDist = 100 ) :selectorSec(mindDist), maxDist(maxDist), maxDistQ(maxDist*maxDist){}

        ssPos get ( const cv::Point2i & p, int scale )
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

//    bool use_lab_format = false;
    ImageSequence< imgtype >  _seq0; //_seq0._lab=use_lab_format;
    ImageSequence< imgtype >  _seq1; //_seq1._lab=use_lab_format;
//    std::vector < ImageSequence<cv::Mat1b> > _occ;

    FlowSequence  _gt;

    int  _cntImages;
    int _patchsize;
    int _scale;

    samplelist _pos;//seq,img, x,y,x2,y2
    samplelist _neg;

    imgtype lpos,lneg,rpos,rneg;
    int _psreal;

	// Add two class variable. sml, ssl
	std::vector<patchselect::selectorMain*>  sml;
	std::vector<patchselect::selectorSec*>  ssl;



    patchselect ( std::string image1, std::string image2, std::string flow, int cntImages, int patchsize = 32, int scale = 1, int offset=0 )
        : _cntImages ( cntImages ),_patchsize ( patchsize ),_scale ( scale )
    {
	    sml.clear();
	    ssl.clear();
	    sml.push_back ( new patchselect::selectorMain() );
	    ssl.push_back ( new patchselect::selectorSec ( 2*scale ) );
	    ssl.push_back ( new patchselect::selectorClose ( 2*scale,200*scale ) );
	    ssl.push_back ( new patchselect::selectorClose ( 2*scale,100*scale ) );
	    ssl.push_back ( new patchselect::selectorClose ( 2*scale,50*scale ) );
	    ssl.push_back ( new patchselect::selectorClose ( 2*scale,20*scale ) );
	    ssl.push_back ( new patchselect::selectorClose ( 2*scale,10*scale ) );
	    ssl.push_back ( new patchselect::selectorClose ( 2*scale,10*scale ) );

        _psreal = patchsize;
        // for ( int i =1; i< scales; i++ ) _psreal*=2;
        //  _psreal*= scale;
//        cout << "Size real:" << _psreal << endl;

        // scaleCut[0]=0;
        // for ( int i = 1, s = _psreal/4;  i< scales ; i++, s/=2 ) scaleCut[i]= scaleCut[i-1] + s;

        // int cnt = 0;

        _seq0.offset = offset;
        _seq1.offset = offset;
        _seq0.setFileSequence ( image1 );
        _seq1.setFileSequence ( image2 );
        _gt.offset = offset;
        _gt._loadType = _gt.LOAD_TYPE_KITTI;
        _gt.setFileSequence ( flow );

        // Load and preprocess seqs.
        for ( int i=0; i< _cntImages; i++ )
        {
            _gt ( i );
            //normalize Patches:
            for ( int k =0; k<2; k++ )
            {
                imgtype x =  k ? _seq0(i) : _seq1(i);
//                cout<< "size of each element is " <<sizeof(x(0,0))<<endl;
                for ( int p=0; p<channels; p++ )
                {
//                        float m_min=10000, m_max=-10000;
//                        for ( int k =0; k<x.rows; k++ )
//                        for ( int l =0; l<x.cols; l++ ){m_min=min(x(k,l)[p], m_min); m_max=max(x(k,l)[p], m_min);};
//                        cout<<"Image "<<i<< " channel " << p<< " Range " << m_min<<" "<<m_max<<endl;
                    double mean = 0;
                    double stdev= 0;

                    for ( int k =0; k<x.rows; k++ )
                        for ( int l =0; l<x.cols; l++ ){
                            stdev += x(k,l)[p] * x(k,l)[p];
                            mean  += x(k,l)[p];
                            }

                    mean/= ( float ) ( x.rows*x.cols );
                    // Actually inverse of std variance.
                    stdev= 1.f/sqrt ( stdev/ ( float ) ( x.rows*x.cols ) );

                    for ( int k =0; k<x.rows; k++ )
                        for ( int l =0; l<x.cols; l++ ){
                        x ( k,l )[p] -= mean;
                        x ( k,l )[p] *= stdev;
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
        if ( _gt[pos.im] ( pos.y,pos.x ) [0] > 1000000.f ) res.x = res.y = -1000;
        return res;
    }



    void reset()
    {
        _pos.resize ( 0 );
        _neg.resize ( 0 );
    }

// cnt: The number of training pairs. (positive pairs, negative pairs). Total num of images = cnt*4
    void add ( int cnt, bool addPositive = true)
    {
//        cv::namedWindow("Image0");
//        cv::namedWindow("Image1");
//        Mat s0,s1;

//        int s =  0;//_scale;//indexx[sfac];
        assert ( _scale ==1 );

        for ( int i =0; i<sml.size(); i++ ) sml[i]->ps = this;

        for ( int i =0; i< cnt; i++ )
        {
            auto pos1 = sml[ randomIVal(sml.size()) ]->get();
            cv::Point2i pos2p = getgtPos2p ( pos1 );

            if ( addPositive && ( pos2p.x<0 || pos2p.y<0 || pos2p.x >= _gt[pos1.im].cols || pos2p.y >= _gt[pos1.im].rows))
                // || _occ[pos1.seq][pos1.im](pos1.y, pos1.x)
            {
                i--;
                continue;
            }
            // cout<<pos2p.y<<" "<< _gt[pos1.im].rows<<endl;
            // assert ( pos2p.y < _gt[pos1.im].rows );

            // float gx = _gt ( pos1.im ) ( pos1.y,pos1.x ) [0];
            // float gy = _gt ( pos1.im ) ( pos1.y,pos1.x ) [0];

            // float gs =  sqrtf ( gx*gx+gy*gy );
            imgtype ref = _seq0[pos1.im] ( cv::Rect ( pos1.x,pos1.y,_psreal,_psreal ) );
            sampletype  n1,p1;


            if ( addPositive )
            {
//                sampletype  p1;
                p1.first = {0, pos1.im, pos1.x, pos1.y, pos2p.x, pos2p.y};
                p1.second[0] = ref;
                // p1.second[0] = _seq0[pos1.im] ( cv::Rect ( pos1.x,pos1.y,_psreal,_psreal ) );
                p1.second[1] = _seq1[pos1.im] ( cv::Rect ( pos2p.x,pos2p.y,_psreal,_psreal ) );

                // cout<<"Break -2"<<endl;

                // imgtype img1 = 
                // imgtype img2 = 
                // cout<<"Break -1"<<endl;


                // cv::resize ( img1 ( cv::Rect ( scaleCut[s], scaleCut[s], img1.cols-scaleCut[s]*2, img1.rows-scaleCut[s]*2 ) ),p1.second[0] ,cv::Size ( _patchsize,_patchsize ),0,0,CV_INTER_AREA );
                // cv::resize ( img2 ( cv::Rect ( scaleCut[s], scaleCut[s], img1.cols-scaleCut[s]*2, img1.rows-scaleCut[s]*2 ) ),p1.second[1] ,cv::Size ( _patchsize,_patchsize ),0,0,CV_INTER_AREA );
                // cv::putText(p1.second[0], std::to_string(s), cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                // cout<<"Break 0"<<endl;
                _pos.push_back ( p1 );
                // assert ( pos1.x >=0 || pos1.y >= 0 && pos2p.x >= 0 && pos2p.y >=0 && ( pos2p.x-pos1.x- _gt[pos1.im] ( pos1.y,pos1.x ) [0] ) <1.1f );
            }

            if ( ssl.size() )   //TODO: set size
            {
                int choosen =  randomIVal ( ssl.size() );
                ssl[choosen]->cols = _gt[pos1.im].cols;
                ssl[choosen]->rows = _gt[pos1.im].rows;
                
                auto pos2n = ssl[choosen]-> get( pos2p, _scale );

//                sampletype  n1;
                float disttopos = sqrtf ( ( pos2n.x - pos2p.x ) * ( pos2n.x - pos2p.x ) + ( pos2n.y - pos2p.y ) * ( pos2n.y - pos2p.y ) );
                n1.first = {disttopos, pos1.im, pos1.x, pos1.y, pos2n.x, pos2n.y};
                n1.second[0] = ref;
                n1.second[1] = _seq1[pos1.im] ( cv::Rect( pos2n.x, pos2n.y, _psreal, _psreal) );
        
                _neg.push_back ( n1 );
                assert ( pos1.x >=0 || pos1.y >= 0 && pos2n.x >= 0 && pos2n.y >=0 && ( pos2n.x-pos1.x- _gt[pos1.im] ( pos1.y,pos1.x ) [0] ) >2.1f );

            }


//            cv::cvtColor(n1.second[0], s0, cv::COLOR_Lab2BGR);
//            cv::cvtColor(n1.second[1], s1, cv::COLOR_Lab2BGR);
//            cv::imshow("Image0", s0);
//            cv::imshow("Image1", s1);
//            cv::waitKey(500);

        }
    }

    void createPyArrayPtr ( float * ptr ) {
    	float * pt=ptr;
    	samplelist* arr =&_pos;
//    	int patch_s_m = _patchsize*_patchsize*channels*sizeof(float);
    	int patch_s   = _patchsize*_patchsize*channels;

    	cout<< "Number of (pos pair, neg pair): " << arr->size()<<endl;
    	for(int i=0; i<(*arr).size(); i++){
    		for(int ss=0; ss<2; ss++){
    			imgtype x = (*arr)[i].second[ss];
    			assert(x.rows == _patchsize);
    			for(int j=0;j<x.rows; j++){
    			    for(int k=0; k<x.cols; k++){
    			        pt[0] = (float) x (j,k) [0];
    			        pt[1] = (float) x (j,k) [1];
    			        pt[2] = (float) x (j,k) [2];
    			        pt += 3;
    			    }


    			}

//    			std::memcpy(pt,(float *)x.data, patch_s_m);
//    			pt +=patch_s;
    		}
    		if (arr == &_pos){
    			arr = &_neg;
    			i--;
    		}
    		else arr = &_pos;
    	}
    	assert(pt == ptr + (patch_s*2*2*arr->size()));

    }

};


