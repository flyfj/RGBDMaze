#pragma once

#include "math.h"

#ifdef _XBOX
#include "xtl.h"
#else
#include "windows.h"
#endif

#ifndef byte
#define byte BYTE
#endif

class ColorConvertHelper
{
public:
	ColorConvertHelper(void);
	~ColorConvertHelper(void);

	//struct
public:
	struct LAB
	{
		double L;
		double A;
		double B;
	};

	struct HSB
	{
		double H;
		double S;
		double B;
	};

	struct XYZ
	{
		double X;
		double Y;
		double Z;
	};

	struct RGB
	{
		int red;
		int green;
		int blue;
	};

	struct CIEXYZ
	{
		double x;
		double y;
		double z;

		CIEXYZ(double dx, double dy, double dz) 
		{
			x = (dx>0.9505)? 0.9505 : ((dx<0)? 0 : dx);
			y = (dy>1.0)? 1.0 : ((dy<0)? 0 : dy);
			z = (dz>1.089)? 1.089 : ((dz<0)? 0 : dz);
		}
	};


	//method
private:
	static double Fxyz(double t)
	{
		return ((t > 0.008856)? pow(t, (1.0/3.0)) : (7.787*t + 16.0/116.0));
	}

public:
    
	static LAB RGBtoLAB(byte red, byte green, byte blue)
	{	
		//RGBtoXYZ		
		// normalize red, green, blue values
		double rLinear = (double)red/255.0;
		double gLinear = (double)green/255.0;
		double bLinear = (double)blue/255.0;

		// convert to a sRGB form
		double r = rLinear;	//(rLinear > 0.04045)? pow((rLinear + 0.055)/(1 + 0.055), 2.2) : (rLinear/12.92) ;
		double g = gLinear;	//(gLinear > 0.04045)? pow((gLinear + 0.055)/(1 + 0.055), 2.2) : (gLinear/12.92) ;
		double b = bLinear;	//(bLinear > 0.04045)? pow((bLinear + 0.055)/(1 + 0.055), 2.2) : (bLinear/12.92) ;

		// converts
		double x = r*0.4124 + g*0.3576 + b*0.1805;
		double y = r*0.2126 + g*0.7152 + b*0.0722;
		double z = r*0.0193 + g*0.1192 + b*0.9505;

		x = (x>0.9505)? 0.9505 : ((x<0)? 0 : x);
		y = (y>1.0)? 1.0 : ((y<0)? 0 : y);
		z = (z>1.089)? 1.089 : ((z<0)? 0 : z);

		//XYZtoLAB
		LAB lab;
		XYZ XYZ_D65;
		XYZ_D65.X = 0.9505;
		XYZ_D65.Y = 1.0;
		XYZ_D65.Z = 1.0890;

		lab.L = 116.0 * Fxyz( y/XYZ_D65.Y ) -16;
		lab.A = 500.0 * (Fxyz( x/XYZ_D65.X ) - Fxyz( y/XYZ_D65.Y) );
		lab.B = 200.0 * (Fxyz( y/XYZ_D65.Y ) - Fxyz( z/XYZ_D65.Z) );

		return lab;
	}

	static CIEXYZ LabtoXYZ(double l, double a, double b)
	{
		double theta = 6.0/29.0;

		double fy = (l+16)/116.0;
		double fx = fy + (a/500.0);
		double fz = fy - (b/200.0);

		XYZ XYZ_D65;
		XYZ_D65.X = 0.9505;
		XYZ_D65.Y = 1.0;
		XYZ_D65.Z = 1.0890;

		return CIEXYZ(
			(fx > theta)? XYZ_D65.X * (fx*fx*fx) : (fx - 16.0/116.0)*3*(theta*theta)*XYZ_D65.X,
			(fy > theta)? XYZ_D65.Y * (fy*fy*fy) : (fy - 16.0/116.0)*3*(theta*theta)*XYZ_D65.Y,
			(fz > theta)? XYZ_D65.Z * (fz*fz*fz) : (fz - 16.0/116.0)*3*(theta*theta)*XYZ_D65.Z
			);
	}	

	static byte Bound(double d)
	{
		return d > 255.0 ? 255 : (d < 0.0? 0 : (byte)d);
	}


	static RGB XYZtoRGB(double x, double y, double z)
	{
		double Clinear[3];
		Clinear[0] = x*3.2410 - y*1.5374 - z*0.4986; // red
		Clinear[1] = -x*0.9692 + y*1.8760 - z*0.0416; // green
		Clinear[2] = x*0.0556 - y*0.2040 + z*1.0570; // blue

		for(int i=0; i<3; i++)
		{
			Clinear[i] = (Clinear[i]<=0.0031308)? 12.92*Clinear[i] : (1+0.055)* pow(Clinear[i], (1.0/2.4)) - 0.055;
		}

		RGB res;
		res.red = Bound(Clinear[0]*255.0);
		res.green = Bound(Clinear[1]*255.0);
		res.blue = Bound(Clinear[2]*255.0);
		return res;
	}

	static RGB LABtoRGB(double l, double a, double b)
	{
		CIEXYZ tmpCIEXYZ( 0, 0, 0 ); 
        tmpCIEXYZ = LabtoXYZ(l, a, b);
		return XYZtoRGB( tmpCIEXYZ.x, tmpCIEXYZ.y, tmpCIEXYZ.z);
	}

	static HSB RGBtoHSB(byte red, byte green, byte blue)
	{	
		double r = ((double)red/255.0);
		double g = ((double)green/255.0);
		double b = ((double)blue/255.0);

		double maxv = max(r,max(g,b));
		double minv = min(r,min(g,b));

		double h = 0.0;
		if(maxv==r && g>=b)
		{
			if(maxv-minv == 0) h = 0.0;
			else h = 60 * (g-b)/(maxv-minv);
		}
		else if(maxv==r && g < b)
		{
			h = 60 * (g-b)/(maxv-minv) + 360;
		}
		else if(maxv == g)
		{
			h = 60 * (b-r)/(maxv-minv) + 120;
		}
		else if(maxv == b)
		{
			h = 60 * (r-g)/(maxv-minv) + 240;
		}

		double s = (maxv == 0)? 0.0 : (1.0-((double)minv/(double)maxv));

		HSB hsb;
		hsb.H = (h>360)? 360 : ((h<0)?0:h); 
		hsb.S = (s>1)? 1 : ((s<0)?0:s);
		hsb.B = (maxv>1)? 1 : ((maxv<0)?0:maxv);
		return hsb;
	}
};