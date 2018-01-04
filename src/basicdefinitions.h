/*
 * File:   basicdefinitions.h
 * Author: bailer
 *
 * Created on 12. April 2012, 20:10
 */


#pragma once

//#include <string>
//#include "OSC.h"
#ifndef _def_no_std
#define _def_std
#endif

#ifdef _def_all
#define _def_std
#define _def_trycatch
#endif


#ifdef _def_std
#define _def_basics
#define _def_timers
#define _def_special_loops
#define _def_assert
#define _def_output
#define _def_stuff
#define def_probability
#endif


//--------------------------BASICS:------------------------------//
#ifdef  _def_basics
#include <iostream>
using std::cout;
using std::endl;
using std::cin;
#endif

//--------------------------LOOPS:------------------------------//
#ifdef _def_special_loops
#define forrange(a,b,c)for( int a=b;a<c;a++)
#define forarea(a1,a2,b1,b2,c1,c2)for(int a1=b1;a1<c1;a1++)for(int a2=b2;a2<c2;a2++)
#define forareass(a1,a2,b1,b2,c1,c2,ss)for(int a1=b1;a1<c1;a1+=ss)for(int a2=b2;a2<c2;a2+=ss)
#define fortimes(a,c)  for( int a=0;a<c;a++)
#define forset(val,a)  for(auto &_x_ : (val))_x_=(a)
#define forclear(val)  for(auto &_x_ : (val))_x_=0
#define forcvmat(a,b,c)  for(int a=0;a<c.rows;a++)for(int b=0;b<c.cols;b++)
//#define forclear(val,a) for( int _i_=0;_i_<a;_i_++)val[_i_]=0
#endif

//--------------------------TRY/CATCH:------------------------------//
#ifdef _def_trycatch
#define autocatch     catch (std::exception e) {   std::cout << "Exception occured: "<< e.what()<<" In line: " <<__LINE__<<" "<<__FILE__<<std::endl;  }
#define autocatchstop     catch (std::exception e) {   std::cout << "Exception occured: "<< e.what()<<" In line: " <<__LINE__<<" "<<__FILE__<< std::endl; exit(0); }
#endif

//--------------------------ASSERT:------------------------------//
#ifdef _def_assert
//Informative assert, shows values:
#define iassert(x,y,z) if(!( (x) y (z))) {std::cerr << "Assertation: "<<  (#x)<<" "<<(#y)<<" "<<(#z)<< " failed! "<< (#x)<<": " << (x)<<" "<<  (#z)<<": " << (z) <<" (Line: "<<__LINE__<<")" <<std::endl; exit(0);}
#define lassert(x)  if(!(x))std::cerr <<"Assertation: " << #x << " failed!"<< " In line: " <<__LINE__<<" "<<__FILE__<< std::endl; // logging assert. Will not crash. Only informative. 
#define assertInRange(f,from,to) assert(!(isnan(f))&& f<=to && f>=from)
//TODO: passert, activate when there is a problem with the code e.g. crashes. 
#ifdef def_test_assert // Test assert, probably slow assertions to test the code 
#define tassert(x)  assert(x)
#else
#define tassert(x)
#endif
#endif


//--------------------------TIMERS:------------------------------//
#ifdef _def_timers
#include <map>
#include <omp.h>



#define __stimeq(x)  x//quiet/deactivated
#define __timeq(s,x) x//quiet/deactivated  

/**
 * Set timer
 * @param s timer handle
 */


/**
 * get timer
 * @param s timer handle
 * @return time
 */

#endif



//--------------------------OUTPUT:------------------------------//

#ifdef _def_output


inline void exitif ( bool condition,std::string what="" )
{
    if ( condition )
    {
        std::cout << what <<" ... exiting"<< std::endl;
        exit ( 0 );
    }
}
inline void printif ( bool condition,std::string what="" )
{
    if ( condition )
    {
        std::cout << what << std::endl;
    }
}

#define _iout(x)  _Pragma("omp critical") std::cout << (#x) << " is \""<< x<<"\" (Line: "<<__LINE__<<")" <<std::endl;
#define _iout2(x,y)  _Pragma("omp critical") std::cout << (#x)<< ", " << (#y) << " is \""<< x<< ", " << y <<"\" (Line: "<<__LINE__<<")" <<std::endl;
#define _iout3(x,y,z)  _Pragma("omp critical") std::cout << (#x)<< ", " << (#y)<< ", " << (#z) << " is \""<< x<< ", " << y<< ", " << z <<"\" (Line: "<<__LINE__<<")" <<std::endl;
#define _ioutn(x) _Pragma("omp critical")  std::cout << (#x) << " is \""<< x <<"\" (Line: "<<__LINE__<<") ";

#define _pout(x) _Pragma("omp critical") std::cout << x << " -> " <<__FUNCTION__ <<"() in line " <<__LINE__<<" "<<__FILE__<<std::endl;
#define _pout2(x,y) _Pragma("omp critical") std::cout << x << " "<< y << " -> " <<__FUNCTION__ <<"() in line " <<__LINE__<<" "<<__FILE__<<std::endl;

#define _outl(x) _Pragma("omp critical") std::cout << x << std::endl
#define _outl2(x,y) _Pragma("omp critical") std::cout << x <<" "<< y << std::endl

#define _out(x) _Pragma("omp critical") std::cout << x << " "
#define _out2(x,y) _Pragma("omp critical") std::cout << x <<" "<< y << " "

#endif

#ifdef def_no_out

#define _iout(x)
#define _iout2(x,y)

#define _ioute(x)
#define _iout2e(x,y)

#define _pout(x)
#define _pout2(x,y)

#define _poute(x)
#define _pout2e(x,y)

#define _out(x)
#define _out2(x,y)

#define _oute(x)
#define _oute2(x,y)

#endif


#ifdef _def_algorithm




#endif

//--------------------------Stuff:------------------------------//

#ifdef _def_stuff

#define _cvMat(y,x) cv::Mat_ < cv::Vec< x, y> >
#define _cvMatf(y) cv::Mat_ < cv::Vec< float, y> >

#define TODO_STOP  std::cout << "Error: Function not imlemented, yet in Line: " <<__LINE__<<"... exiting"<< std::endl; exit(0);
#define TODO_WARN  std::cout << "Warning: Function not imlemented, yet in Line: " <<__LINE__<< std::endl;

#endif





//exotic:
 //REQUIRES CPP FILE TO WORK: //
#ifdef def_probability

#include <boost/random.hpp>

struct _DEF_PROB
{
    static  boost::mt19937 _rng;
    static  boost::uniform_real<float> _rf;

    inline static float get()
    {
        return _rf(_rng);
    }

};

#define withprobability(x)     if(_DEF_PROB::get()<x)
#define TRUE_WITH_PROBABILITY(x)  (_DEF_PROB::get()<x)
inline float randomprobability()
{
    return _DEF_PROB::get();
}

inline float randomVal(float max)
{
    return _DEF_PROB::get()*max;
}

inline int randomIVal(float max)
{
    return int(_DEF_PROB::get()*max);
}
#endif
