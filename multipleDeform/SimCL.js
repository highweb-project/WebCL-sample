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

// local OpenCL info
var platforms;                           // array of OpenCL platform ids
var platform;                            // OpenCL platform id
var devices;                             // array of OpenCL device ids
var device;                              // OpenCL device id
var context;                             // OpenCL context
var queue;                               // OpenCL command queue
var program;                             // OpenCL program
var kernel;                              // OpenCL kernel

var initPosBuffer;                       // OpenCL buffer
var curPosBuffer;                        // OpenCL buffer created from GL VBO
var curNorBuffer;                        // OpenCL buffer created from GL VBO

var globalWorkSize = new Int32Array(2);
var localWorkSize = new Int32Array(2);

function getKernel(id) {
    var kernelScript = document.getElementById(id);
    if (kernelScript === null || kernelScript.type !== "x-kernel")
        return null;

    return kernelScript.firstChild.textContent;
}

function InitCL() {
    try  {
        if (typeof(webcl) === "undefined") {
            console.error("webcl property is yet to be defined in window.");
            return null;
        }

        var cl = webcl;
        if (cl === null) {
            console.error("Failed to fetch a webcl instance.");
            return null;
        }

        // Select a compute device
        platforms = cl.getPlatforms();
        if (platforms.length === 0) {
            console.error("No platforms available");
            return null;
        }
        platform = platforms[0];
        // Select a compute device
        devices = platform.getDevices(cl.DEVICE_TYPE_GPU);
        if (devices.length === 0) {
            console.error("No devices available");
            return null;
        }
        device = devices[0];
	console.log("before enableExtension: GLCL_SHARE_MODE = " + GLCL_SHARE_MODE);
        if (GLCL_SHARE_MODE) {
            //GLCL_SHARE_MODE = cl.enableExtension("KHR_GL_SHARING");
	    try { 
	        GLCL_SHARE_MODE = cl.enableExtension("KHR_gl_sharing");		
	    } catch (e) {
		console.log("WEBCLGL: sharing WebGL resources --- " + e.message);
	    }	
	}
	console.log("after enableExtension: GLCL_SHARE_MODE = " + GLCL_SHARE_MODE);
	
	GLCL_SHARE_MODE = false; // for test only 
        // Create a compute context
        if (GLCL_SHARE_MODE) {
	    try {
                context = cl.createContext(userData.gl, devices);
	    } catch (e) {
		console.error("WEBCLGL: " + e.message);
	    }
	}
        else
            context = cl.createContext(devices);
        if(context === null) {
            console.error("createContext fails");
            return null;
        }

        // Create a command queue
        queue = context.createCommandQueue(device);

        // Create the compute program from the source buffer
        var kernelSource = getKernel("deform_kernel");
        if (kernelSource === null) {
            console.error("No kernel named: " + "deform_kernel");
            return null;
        }

        program = context.createProgram(kernelSource);

        // Build the program executable
        program.build([device]);

        // Create the compute kernel in the program we wish to run
        kernel = program.createKernel("displace");
    } catch (e) {
        console.error("Deform Demo Failed ; Message: " + e.message);
    }
    return cl;
}

function divide_up(a, b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

function InitCLBuffers(cl) {
    try {
        if (cl === null)
            return;

        var bufferSize = userData.nVertices * NUM_VERTEX_COMPONENTS * Float32Array.BYTES_PER_ELEMENT;

        // Create CL working buffers
        initPosBuffer = context.createBuffer(cl.MEM_WRITE_ONLY, bufferSize);
        if (initPosBuffer === null) {
            console.error("Failed to allocate device memory");
            return null;
        }

        // Create CL buffers from GL VBOs
        // (Initial load of positions is via gl.bufferData)
	console.log("GLCL_SHARE_MODE = " + GLCL_SHARE_MODE);
        if (GLCL_SHARE_MODE) 
            curPosBuffer = context.createFromGLBuffer(cl.MEM_READ_WRITE, userData.curPosVBO);
        else 
            curPosBuffer = context.createBuffer(cl.MEM_READ_WRITE, bufferSize);
        if (curPosBuffer === null) {
            console.error("Failed to allocate device memory");
            return null;
        }
	console.log("InitCLBuffer() : GLCL_SHARE_MODE = " + GLCL_SHARE_MODE);
        if (GLCL_SHARE_MODE) 
            curNorBuffer = context.createFromGLBuffer(cl.MEM_READ_WRITE, userData.curNorVBO);
        else 
            curNorBuffer = context.createBuffer(cl.MEM_READ_WRITE, bufferSize);
        if (curNorBuffer === null) {
            console.error("Failed to allocate device memory");
            return null;
        }

        // Get the maximum work group size for executing the kernel on the device
        //
        var workGroupSize = kernel.getWorkGroupInfo(device, cl.KERNEL_WORK_GROUP_SIZE);

        globalWorkSize[0] = 1;
        globalWorkSize[1] = 1;
        while(globalWorkSize[0] * globalWorkSize[1] <userData.nVertices) {
            globalWorkSize[0] = globalWorkSize[0] * 2;
            globalWorkSize[1] = globalWorkSize[1] * 2;
        }

        localWorkSize[0] = globalWorkSize[0];
        localWorkSize[1] = globalWorkSize[1];
        while (localWorkSize[0] * localWorkSize[1] > workGroupSize) {
            localWorkSize[0] = localWorkSize[0] / 2;
            localWorkSize[1] = localWorkSize[1] / 2;
        }
	

        console.log("workGroupSize: " + workGroupSize);
        console.log("localWorkSize[0]: " + localWorkSize[0]);
        console.log("localWorkSize[1]: " + localWorkSize[1]);
        console.log("globalWorkSize[0]: " + globalWorkSize[0]);
        console.log("globalWorkSize[1]: " + globalWorkSize[1]);

        // Initial load of initial position data
        queue.enqueueWriteBuffer(initPosBuffer, true, 0, bufferSize, userData.initPos);

        queue.finish();
    }
    catch (e) {
        console.error("Deform Demo Failed ; Message: " + e.message);
    }
}

function SimulateCL(cl)
{
  try {
    if (cl === null)
      return;

    if (GLCL_SHARE_MODE) {
      queue.enqueueAcquireGLObjects([curNorBuffer]);
      queue.enqueueAcquireGLObjects([curNorBuffer]);
    }

    kernel.setArg(0, initPosBuffer);
    kernel.setArg(1, curNorBuffer);
    kernel.setArg(2, curPosBuffer);
    kernel.setArg(3, new Float32Array([userData.frequency]));
    kernel.setArg(4, new Float32Array([userData.amplitude]));
    kernel.setArg(5, new Float32Array([userData.phase]));
    kernel.setArg(6, new Float32Array([userData.lacunarity]));
    kernel.setArg(7, new Float32Array([userData.increment]));
    kernel.setArg(8, new Float32Array([userData.octaves]));
    kernel.setArg(9, new Float32Array([userData.roughness]));
    kernel.setArg(10, new Uint32Array([userData.nVertices]));

    queue.enqueueNDRangeKernel(kernel, globalWorkSize.length, [], globalWorkSize, localWorkSize);
    //queue.enqueueNDRangeKernel(kernel, globalWorkSize.length, [0,0], [globalWorkSize[0],globalWorkSize[1]], [localWorkSize[0],localWorkSize[1]]);
    queue.finish();

    if (GLCL_SHARE_MODE) {
      queue.enqueueReleaseGLObjects([curPosBuffer]);
      queue.enqueueReleaseGLObjects([curNorBuffer]);
    } else {
      var bufferSize = userData.nVertices * NUM_VERTEX_COMPONENTS * Float32Array.BYTES_PER_ELEMENT;
      queue.enqueueReadBuffer(curPosBuffer, true, 0, bufferSize, userData.curPos );
      queue.enqueueReadBuffer(curNorBuffer, true, 0, bufferSize, userData.curNor);
    }
  }
  catch (e) {
    console.error("Deform Demo Failed ; Message: " + e.message);
  }

  userData.phase += PHASE_DELTA;
}
