#include "patchselect.h"
#include "torch.h"
#include <algorithm>

lua::State s;

int _samples;
int _patchsize;
patchselect * _ps;

std::vector<patchselect::selectorMain*>  sml;
std::vector<patchselect::selectorSec*>  ssl;


std::vector < std::array<short,3> > outlierIDs;
int i_id = 0;


const int channels =3;

static int requestNewData ( lua_State *L )
{
    lua::State l ( L );
    int num = l.Pop<int>();

    bool _50 = false;
    if ( num<0 )
    {
        num = -num;
        _50 = true;
    }

    //cout << "Data request of size: " << num << endl;
    _ps->reset();
    _ps->add ( num/2, sml,ssl,true,_50 );

    assert ( num == _ps->_pos.size() + _ps->_neg.size() );

    THFloatTensor * x = THFloatTensor_newWithSize1d ( num*2*channels*_patchsize*_patchsize );

    if ( false && patchselect::scales>1 )
    {
        long size[6] = {num,2, patchselect::scales,channels,_patchsize,_patchsize };
        THLongStorage * xsize = THLongStorage_newWithData ( size,6 );
        THFloatTensor_resize ( x, xsize,  NULL );
    }
    else
    {
        long size[5] = {num,2,channels,_patchsize,_patchsize };
        THLongStorage * xsize = THLongStorage_newWithData ( size,5 );
        THFloatTensor_resize ( x, xsize,  NULL );
    }
    float* y = THFloatTensor_data ( x );
    _ps->createTorchPtr ( y, 2 );
    luaT_pushudata ( L, x, "torch.FloatTensor" );
    //l.Push(x);
    return true;
}

static int getData ( lua_State *L )
{
    int samples = _ps->_pos.size() + _ps->_neg.size();
    //cout << "Data of size: " << samples << endl;
    THFloatTensor * x = THFloatTensor_newWithSize1d ( samples*2*channels*_patchsize*_patchsize );

    if ( false && patchselect::scales>1 )
    {
        long size[6] = {samples,2, patchselect::scales,channels,_patchsize,_patchsize };
        THLongStorage * xsize = THLongStorage_newWithData ( size,6 );
        THFloatTensor_resize ( x, xsize,  NULL );
    }
    else
    {
        long size[5] = {samples,2,channels,_patchsize,_patchsize };
        THLongStorage * xsize = THLongStorage_newWithData ( size,5 );
        THFloatTensor_resize ( x, xsize,  NULL );
    }
    // THFloatTensor_resize5d ( x, _samples*2,2* patchselect::scales,3,_patchsize,_patchsize );
    float* y = THFloatTensor_data ( x );
    _ps->createTorchPtr ( y, 2 );
    luaT_pushudata ( L, x, "torch.FloatTensor" );
    return 1;
}


static int loadDist ( lua_State *L )
{
    THFloatTensor * x = THFloatTensor_newWithSize1d ( _ps->_pos.size() + _ps->_neg.size() );
    float* y = THFloatTensor_data ( x );
    _ps->createTorchPtrDist ( y );
    luaT_pushudata ( L, x, "torch.FloatTensor" );
    return 1;
}

static int loadInfo ( lua_State *L )
{
    THFloatTensor * x = THFloatTensor_newWithSize3d ( _ps->_pos.size() + _ps->_neg.size(),3, 2 );
    float* y = THFloatTensor_data ( x );
    _ps->createTorchPtrInfo ( y );
    luaT_pushudata ( L, x, "torch.FloatTensor" );
    return 1;
}

int lib_init ( lua_State *L )
{
    lua::State l ( L );
    double scale =  l.Pop<double>();
    int size =  l.Pop<int>();
    int start =  l.Pop<int>();
    _patchsize = l.Pop<int>();
    cout << "scale (not forced to 1):"<< scale << " start: " << start<< " size: " << size << " patchsize: " << _patchsize << endl;

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
    // ssl.push_back ( new patchselect::selectorClose ( 2*scale,6*scale ) );
    //  ssl.push_back ( new patchselect::selectorClose ( 2*scale,4*scale ) );
    //scale = 1;

    if ( _ps ) delete _ps;

    //  _ps =  new patchselect ("/bailerdata/kitti2015/training/image_2/%6_10.png","/bailerdata/kitti2015/training/image_2/%6_11.png", "/bailerdata/kitti2015/training/flow_noc/%6_10.png", size,_patchsize,scale,start );
    _ps =  new patchselect ( "/bailerdata/kitti/training/colored_0/%6_10.png","/bailerdata/kitti/training/colored_0/%6_11.png", "/bailerdata/KITTI/training/flow_noc/%6_10.png", size,_patchsize,scale,start );
    // _ps =  new patchselect ("/bailerdata/kitti/training/colored_0/%6_10.png","/bailerdata/kitti/training/colored_0/%6_11.png", "/bailerdata/kitti/training/output2/%6_10.flo", size,_patchsize,scale,start );
}

extern "C"
{
    int testen ( lua_State *L )
    {
        lua_pushnumber ( L, 1 );
        return 1;
    }

    static const struct luaL_reg mylib [] =
    {
        {"loadDist",loadDist},{"newData", requestNewData},{"test", testen},{"getInfo", loadInfo},{ "init",lib_init }, {NULL, NULL}  /* sentinel */
    };
    int luaopen_libpatchtrain ( lua_State *L )
    {
        luaL_openlib ( L, "mylib", mylib, 0 );
        _ps = 0;
        return 1;
    }
}



int main()
{
    /* s.reg ( "requestNewData", requestNewData );
     s.reg ( "getInfo", loadInfo );
     s.reg ( "getData", getData );
     s.reg ( "lib_init", lib_init );

     s.Call<void> ( "lib_init",32,2,1 );

     _ps->add ( 100000, sml,ssl,true,false );
     _ps->cmpCoop();

      exit ( 0 );*/
    _patchsize= 64;//38
    _samples = 100; //

    patchselect ps ( "/bailerdata/kitti/training/colored_0/%6_10.png","/bailerdata/kitti/training/colored_0/%6_11.png", "/bailerdata/KITTI/training/flow_noc/%6_10.png", 1,_patchsize, 1,5 );

    sml.push_back ( new patchselect::selectorMain() );
    ssl.push_back ( new patchselect::selectorClose ( 20,21 ) );
    //  ssl.push_back ( new patchselect::selectorClose ( 8,100 ) );
    //  ssl.push_back ( new patchselect::selectorClose ( 8,20 ) );

    ps.add ( _samples*10, sml,ssl,true,false );

    ps.reset();
    ps.add ( _samples, sml,ssl,true,false );
    //ps.cmpCoop();

    /*
      ps.cmpTemplate(1);
      ps.cmpTemplate(2);
      ps.cmpTemplate(4);

      ps.cmpTemplate(6);
      ps.cmpTemplate(7);*/



    //ps.cmpTemplate();
    //exit(0);
    ps.createDbgImgages ( 10 );

    cv::imshow ( "lpos",ps.lpos );
    cv::imshow ( "rpos",ps.rpos );

    cv::imshow ( "lneg",ps.lneg );
    cv::imshow ( "rneg",ps.rneg );
    cv::waitKey ( 0 );
    exit ( 0 );
    //exit ( 0 );

    _ps = &ps;

    s.Load ( "../lua/main.lua" );

    exit ( 0 );

}
