import numpy, sys, os
from pyxtractor import pyx
import itertools
from scipy.interpolate import LinearNDInterpolator as interp
from scipy.optimize import fmin, minimize, anneal
from scipy.spatial import KDTree
import pyfits

def ifilter_mag(iterable):
    ##-- filter only star matches that are within 0.25 mags of each other
    for x in iterable:
        if abs( x[0][2] - x[1][2] ) < 0.25:
            yield x[0:2]

def match( xv1, xv2 ):
    ##-- a cross-correlation matcher that assumes
    ##-- that there is no rotation or scale variation
    ##-- between images.
    
    ilist = ifilter_mag( itertools.product( xv1, xv2 ) )
    dxy = map( lambda x: [ x[0][0] - x[1][0], x[0][1] - x[1][1] ], ilist )
    
    dxy = numpy.asarray( dxy ).T    
    tree = KDTree( dxy.T )
    tree_xy = KDTree( xv1[:,0:2] )
    print 'trees built - correlating...'
    
    N = map( lambda x: len( tree.query_ball_point( x, 5.0 ) ), dxy.T )
    x = numpy.argmax(N)

    xv1ret, xv2ret = [],[]
    for k in xv2[:,0:2]:
        r, i = tree_xy.query(  [ k[0] + dxy.T[x][0], k[1] + dxy.T[x][1] ] )
        if r < 5:
            xv1ret.append( xv1[i][0:2] )
            xv2ret.append( k )
    
    return numpy.asarray(xv1ret), numpy.asarray(xv2ret)


def get_data( ims ):
    ##-- Uses Pyxtractor (source extractor plugin) to extract high-sig
    ##-- sources from two images.
    
    T = pyx()
    T.params = ['XWIN_IMAGE', 'YWIN_IMAGE', 'MAG_AUTO', 'FLAGS', 'FWHM_IMAGE']
    T.options[ 'DETECT_THRESH' ] = 500
    T.getcat( ims )
    T.cleanup()
    xy1, xy2 = [],[]

    OK1 = numpy.where( (T.catalog[ims[0]]['FLAGS']<4)*(T.catalog[ims[0]]['FWHM_IMAGE'] > 3) )
    OK2 = numpy.where( (T.catalog[ims[1]]['FLAGS']<4)*(T.catalog[ims[1]]['FWHM_IMAGE'] > 3) )

    xv1 = numpy.asarray( [ T.catalog[ims[0]]['XWIN_IMAGE'][OK1],
                           T.catalog[ims[0]]['YWIN_IMAGE'][OK1],
                           T.catalog[ims[0]]['MAG_AUTO'][OK1] ]).T
    xv2 = numpy.asarray( [ T.catalog[ims[1]]['XWIN_IMAGE'][OK2],
                           T.catalog[ims[1]]['YWIN_IMAGE'][OK2],
                           T.catalog[ims[1]]['MAG_AUTO'][OK2]] ).T
        
    return match( xv1, xv2 )
     
    

def remap( Xi, Yi, params):
    X = Xi - max(Xi)*0.5
    Y = Yi - max(Yi)*0.5

    ZXx = params[0:3]
    ZXy = params[3:6]
    ZYx = params[6:9]
    ZYy = params[9:12]

    ##-- no cross terms yet!
            
    xin = numpy.poly1d(ZXx)(X) + numpy.poly1d(ZXy)(Y)
    yin = numpy.poly1d(ZYx)(X) + numpy.poly1d(ZYy)(Y)
    return xin + max(Xi)*0.5, yin + max(Yi)*0.5


def interpmap( xsamp, ysamp, xin, yin, zref ):
    i = interp( numpy.asarray( [xin, yin] ).T, zref, fill_value=0.0 )
    return i( numpy.asarray( [xsamp, ysamp] ).T )


def ffunc( params, x1, y1, x2, y2 ):

    xf, yf = remap( x1, y1, params )
    dr = numpy.sort( (xf - x2)**2 + (yf-y2)**2 )
    ### crude outlier rejection: remove worst 2 points  
    return numpy.mean( dr[:-2] )


def computemap( xstars1, ystars1, xstars2, ystars2 ):
    dx = numpy.mean(xstars2 - xstars1)
    dy = numpy.mean(ystars2 - ystars1)
    
    params0 = [ 0.0, 1.0, dx,  ##-- Xx
                0.0, 0.0, 0.0, ##-- Xy
                0.0, 0.0, 0.0, ##-- Yx
                0.0, 1.0, dy ] ##-- Yy

    print 'optimizing...'
    params1 = fmin( ffunc, params0, args=( xstars1, ystars1, xstars2, ystars2, ), maxiter=1000 )
    
    print
    print 'Solution:'
    print 'scale     X %.4f Y %.4f'%( params1[1], params1[10] )
    print 'offset    X %.4f Y %.4f'%( params1[2], params1[11] )
    print 'start off X %.4f Y %.4f'%( dx, dy )
    print
    print
    
    return params1


def warp2ref( imlist ):
    
    for im in imlist:

        ##-- open image to be matched
        h1 = pyfits.open(im)
        orig_im = h1[0].data
        h1.close()

        os.system('cp %s %s_masksubmedian.fits'%(im, im))


        delta_arr = []
        mask_arr = []
        for im2 in imlist:
            if im2 == im:
                continue

            print
            print 'Preparing to warp %s to %s...'%(im2, im)
            ##-- open image to be warped
            h1 = pyfits.open(im2)
            a = h1[0].data
            h1.close()
            
            b = numpy.indices( a.shape ) + 1 ##-- source extractor convention?
            mv = a.shape

            ##-- zeroing edge pixel to mask edges in interpolation
            a[b[0] == 1] = 0
            a[b[1] == 1] = 0
            a[b[0] == mv[0]] = 0
            a[b[1] == mv[1]] = 0
            
            ##-- collect matched star list 
            xva, xvb = get_data( [im2, im] )

            ##-- fit warp to matched star list
            params = computemap( xva.T[0], xva.T[1], xvb.T[0], xvb.T[1])

            ##-- warp image
            print 'Warping...'
            Y, X = b[0].ravel(), b[1].ravel()
            X2, Y2 = remap( X, Y, params=params)

            V2 = interpmap( X, Y, X2, Y2, a.ravel() )
            V3 = numpy.reshape( V2, a.shape )

            ##-- compute difference image and mask regions
            diff_im = orig_im - V3
            mask1 = numpy.where( V3 == 0.0 )
            mask2 = numpy.where( V3 > 0.0 )

            ##-- fix any obvious offsets.
            delta_median =  numpy.median( diff_im[mask2] )
            print 'MEDIAN SKY OFFSET = %.1f counts'%(delta_median)
            
            diff_im -= delta_median
            diff_im[mask1] = 0.0
            
            ##-- write out corrected difference
            #hdu = pyfits.PrimaryHDU( diff_im - delta_median )
            #hdu.writeto( '%s_minus_%s.fits'%(im, im2) )

            #print '%s_minus_%s.fits written.'%(im, im2)

            delta_arr.append( diff_im - delta_median )
            mask_arr.append( V3 >= 0.0 )
            
        delta_arr = numpy.asarray( delta_arr )
        mask_arr = numpy.asarray( mask_arr )

        masked_data = numpy.ma.asarray( delta_arr, mask=mask_arr )
        final_im = numpy.ma.MaskedArray.filled( numpy.ma.median( masked_data, axis=0 ), 0 )

        h2 = pyfits.open('%s_masksubmedian.fits'%(im), mode='update')
        h2[0].data = final_im
        h2.flush()

imlist = ['tt1.fits', 'tt2.fits', 'tt3.fits','tt4.fits', 'tt5.fits']

warp2ref( imlist )
