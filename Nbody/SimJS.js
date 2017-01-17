/*
* Copyright (C) 2011 Samsung Electronics Corporation. All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided the following conditions
* are met:
* 
* 1.  Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
* 
* 2.  Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY SAMSUNG ELECTRONICS CORPORATION AND ITS
* CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING
* BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SAMSUNG
* ELECTRONICS CORPORATION OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS, OR BUSINESS INTERRUPTION), HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
* NEGLIGENCE OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
 
function SimulateJS() {
    var curPos = userData.curPos;
    var curVel = userData.curVel;
    var nxtPos = userData.nxtPos;
    var nxtVel = userData.nxtVel;
    
 
    var x, y, z, m;             // position and mass particle i
    var vx, vy, vz;             // velocity particle i
    var ax, ay, az;             // acceleration particle i
    var xj, yj, zj, mj;         // position and mass of particle j
    var rx, ry, rz;             // distance between particle i and particle j
    var distSqr;                // square of distance between particle i and particle j
    var invDist, invDistCube;   // inverse distance between particle i and particle j
    var s;                      // mass * invDistCube
    
    var ii;						// index for particle i
    var jj;						// index for particle j
    
    for(var i=0; i<NBODY; i++) {
        ax = ay = az = 0;       
        ii = 4*i;
        
        x = curPos[ii + 0];
        y = curPos[ii + 1];
        z = curPos[ii + 2];
        m = curPos[ii + 3];
        
        vx = curVel[ii + 0];
        vy = curVel[ii + 1];
        vz = curVel[ii + 2];

        for(var j=0; j<NBODY; j++) {       
        	jj = 4*j;
        	
            xj = curPos[jj + 0];
            yj = curPos[jj + 1];
            zj = curPos[jj + 2];
            mj = curPos[jj + 3];

			rx = (xj - x);
			ry = (yj - y);
			rz = (zj - z);

			distSqr = rx*rx + ry*ry + rz*rz;
            invDist = 1 / Math.sqrt(distSqr + EPSSQR);
			invDistCube = invDist * invDist * invDist;
			s = mj * invDistCube;

			ax += (s*rx);
			ay += (s*ry);
			az += (s*rz);  
        }
        
        nxtPos[ii + 0] = x + (vx * DT) + (0.5 * ax * DT * DT);
        nxtPos[ii + 1] = y + (vy * DT) + (0.5 * ay * DT * DT);
        nxtPos[ii + 2] = z + (vz * DT) + (0.5 * az * DT * DT);
        nxtPos[ii + 3] = m;
        
        nxtVel[ii + 0] = vx + (ax * DT);
        nxtVel[ii + 1] = vy + (ay * DT);
        nxtVel[ii + 2] = vz + (az * DT);
        
        CheckBoundry(i);
    }
    
    // now "flip" cur and nxt
    
    userData.curPos = nxtPos;
    userData.curVel = nxtVel;
    userData.nxtPos = curPos;
    userData.nxtVel = curVel;
}

// if we go outside, reincarnate near center
 //
function CheckBoundry(i) {
    var nxtPos = userData.nxtPos;
    var nxtVel = userData.nxtVel;
    var ii = 4*i;
    var x = nxtPos[ii + 0];
    var y = nxtPos[ii + 1];
    var z = nxtPos[ii + 2];
    
    if(x > 1 || x < -1 || y > 1 || y < -1 || z > 1 || z < -1) {    
        var r = 0.1 *  RAND0TO1();
	    var theta = Math.PI * RAND0TO1();
        var phi = 2 * Math.PI * RAND0TO1();
        nxtPos[ii + 0] = r * Math.sin(theta) * Math.cos(phi);
        nxtPos[ii + 1] = r * Math.sin(theta) * Math.sin(phi);
        nxtPos[ii + 2] = r * Math.cos(theta);
        nxtVel[ii + 0] = 0;
        nxtVel[ii + 1] = 0;
        nxtVel[ii + 2] = 0;
    }
}
