var scale = [];
var corner = [];

function ScalarField(dim, viscosity, dt, boundaries, box) {
    var i, j, k;

    this.field = new Float32Array(getFlatSize(dim));
    this.box = box;

    this.dim = dim;
    this.viscosity = viscosity;
    this.dt = dt;

    scale = [1 / (this.dim + 2) * 2, 1 / (this.dim + 2) * 2, 1 / (this.dim + 2) * 2];
    corner = [-(this.dim + 2) / 2, -(this.dim + 2) / 2, -(this.dim + 2) / 2];

    for (i = 0; i < this.dim + 2; i++) {
        for (j = 0; j < this.dim + 2; j++) {
            for (k = 0; k < this.dim + 2; k++) {
                this.field[index(i, j, k, dim)] = 0.0;
            }
        }
    }

  // WebGL texture setup

    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  // WebGL shader setup

    this.shaderProgram2D = simpleSetup(gl, "2d-vertex-shader", "2d-fragment-shader", ["a_position", "a_texCoord"], [ 0, 0, 0, 0 ], 10000);
    this.shaderProgram2D.positionLocation = gl.getAttribLocation(this.shaderProgram2D, "a_position");
    this.shaderProgram2D.texCoordLocation = gl.getAttribLocation(this.shaderProgram2D, "a_texCoord");
    this.shaderProgram2D.resolutionLocation = gl.getUniformLocation(this.shaderProgram2D, "u_resolution");
    gl.uniform2f(this.shaderProgram2D.resolutionLocation, canvas.width, canvas.height);

  // WebGL vertex attrib setup

    var texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        0.0,  0.0,
        1.0,  0.0,
        0.0,  1.0,
        0.0,  1.0,
        1.0,  0.0,
        1.0,  1.0]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.shaderProgram2D.texCoordLocation);
    gl.vertexAttribPointer(this.shaderProgram2D.texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(this.shaderProgram2D.positionLocation);
    gl.vertexAttribPointer(this.shaderProgram2D.positionLocation, 2, gl.FLOAT, false, 0, 0);

    (function setRectangle(gl, x, y, width, height) {
        var x1 = x;
        var x2 = x + width;
        var y1 = y;
        var y2 = y + height;
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            x1, y1,
            x2, y1,
            x1, y2,
            x1, y2,
            x2, y1,
            x2, y2]), gl.STATIC_DRAW);
    }(gl, 0, 0, canvas.width, canvas.height));
}

ScalarField.prototype.setTimestep = function (value) {
    this.dt = value;
};

ScalarField.prototype.setViscosity = function (value) {
    this.viscosity = value;
};

ScalarField.prototype.reset = function () {
    var bufSize = 4 * numCells, i, j, k;

    for (i = 0; i < this.dim + 2; i++) {
        for (j = 0; j < this.dim + 2; j++) {
            for (k = 0; k < this.dim + 2; k++) {
                this.field[index(i, j, k, dim)] = 0.0;
            }
        }
    }

    try {
        clQueue.enqueueWriteBuffer(scalarBuffer, false, 0, bufSize, this.getField(), []);
    } catch (e) {
        console.error("scalarField.reset", [e]);
    }
};

ScalarField.prototype.draw = function (viewer) {

    volumeRayMarchingKernel.setArg(0, pixelBuffer);
    volumeRayMarchingKernel.setArg(1, scalarBuffer);
    volumeRayMarchingKernel.setArg(2, new Int32Array([canvas.width]));
    volumeRayMarchingKernel.setArg(3, new Int32Array([canvas.height]));
    volumeRayMarchingKernel.setArg(4, new Float32Array([((canvas.height / 2) / Math.tan(Math.PI / 8))]));
    volumeRayMarchingKernel.setArg(5, new Float32Array([-cubePos]));
    volumeRayMarchingKernel.setArg(6, new Float32Array([2.0]));
    volumeRayMarchingKernel.setArg(7, new Int32Array([this.dim]));
    volumeRayMarchingKernel.setArg(8, new Float32Array([ds]));

    try {
        var globalWS = new Int32Array(2);
        globalWS[0] = Math.ceil(canvas.width / 32) * 32;
        globalWS[1] = Math.ceil(canvas.height / 32) * 32;
        var start = Date.now();
        clQueue.enqueueNDRangeKernel(volumeRayMarchingKernel, globalWS.length, [], globalWS, []);
        clQueue.enqueueReadBuffer(pixelBuffer, true, 0, pixelCount * 4, pixels, []);
        raymarchTime += Date.now() - start;
    } catch (e) {
        console.error("scalarField.draw", [e]);
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    gl.useProgram(this.shaderProgram2D);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};

ScalarField.prototype.getField = function () {
    return this.field;
};

ScalarField.prototype.step = function (source) {
    var bufSize = 4 * numCells;
    this.addField(source);
    this.diffusion();
    this.advection();

    var start = Date.now();
    try {
        clQueue.enqueueReadBuffer(scalarBuffer, true, 0, bufSize, this.field);
    } catch (e) {
        console.error("scalarField.step", [e]);
    }

    clMemTime = Date.now() - start;
};

ScalarField.prototype.addField = function (source) {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        scalarAddKernel.setArg(0, scalarBuffer);
        scalarAddKernel.setArg(1, scalarSourceBuffer);
        scalarAddKernel.setArg(2, new Uint32Array([this.dim]));
        scalarAddKernel.setArg(3, new Float32Array([this.dt]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(scalarAddKernel, globalWS.length, [], globalWS, []);
        clQueue.finish();
        clTime += Date.now() - start;
    } catch (e) {
        console.error("scalarField.addField", [e]);
    }
};

ScalarField.prototype.diffusion = function () {
    var globalWS = new Int32Array(3);
    var i;
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        scalarCopyKernel.setArg(0, scalarBuffer);
        scalarCopyKernel.setArg(1, scalarTempBuffer);
        scalarCopyKernel.setArg(2, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(scalarCopyKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        for (i = 0; i < 20; i++) {
            scalarDiffusionKernel.setArg(0, scalarBuffer);
            scalarDiffusionKernel.setArg(1, scalarTempBuffer);
            scalarDiffusionKernel.setArg(2, new Uint32Array([this.dim]));
            scalarDiffusionKernel.setArg(3, new Float32Array([this.dt]));
            scalarDiffusionKernel.setArg(4, new Float32Array([this.viscosity]));

            start = Date.now();
            clQueue.enqueueNDRangeKernel(scalarDiffusionKernel, globalWS.length, [], globalWS, []);
            clTime += Date.now() - start;
        }
    } catch (e) {
        console.error("scalarField.diffusion", [ e]);
    }

    this.setBoundaryDensities();
};

ScalarField.prototype.advection = function () {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        scalarCopyKernel.setArg(0, scalarBuffer);
        scalarCopyKernel.setArg(1, scalarTempBuffer);
        scalarCopyKernel.setArg(2, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(scalarCopyKernel, globalWS.length, [], globalWS, []);
        //clQueue.finish();
        clTime += Date.now() - start;

        scalarAdvectionKernel.setArg(0, scalarBuffer);
        scalarAdvectionKernel.setArg(1, scalarTempBuffer);
        scalarAdvectionKernel.setArg(2, vectorBuffer);
        scalarAdvectionKernel.setArg(3, new Uint32Array([this.dim]));
        scalarAdvectionKernel.setArg(4, new Float32Array([this.dt]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(scalarAdvectionKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("scalarField.advection", [e]);
    }

    this.setBoundaryDensities();
};

ScalarField.prototype.setBoundaryDensities = function () {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        scalarBoundariesKernel.setArg(0, scalarBuffer);
        scalarBoundariesKernel.setArg(1, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(scalarBoundariesKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("scalarField.setBondaryDensities", [e]);
    }
};

ScalarField.prototype.setCornerDensities = function () {
    this.field[index(0, 0, 0, dim)] = (this.field[index(1, 0, 0, dim)] + this.field[index(0, 1, 0, dim)] + this.field[index(0, 0, 1, dim)]) / 3;
    this.field[index(0, this.dim + 1, 0, dim)] = (this.field[index(1, this.dim + 1, 0, dim)] + this.field[index(0, this.dim, 0, dim)] + this.field[index(0, this.dim + 1, 1, dim)]) / 3;
    this.field[index(this.dim + 1, 0, 0, dim)] = (this.field[index(this.dim, 0, 0, dim)] + this.field[index(this.dim, 1, 0, dim)] + this.field[index(this.dim + 1, 0, 1, dim)]) / 3;
    this.field[index(this.dim + 1, this.dim + 1, 0, dim)] = (this.field[index(this.dim, this.dim + 1, 0, dim)] + this.field[index(this.dim + 1, this.dim, 0, dim)] + this.field[index(this.dim + 1, this.dim + 1, 1, dim)]) / 3;
    this.field[index(0, 0, this.dim + 1, dim)] = (this.field[index(1, 0, this.dim + 1, dim)] + this.field[index(0, 1, this.dim + 1, dim)] + this.field[index(0, 0, this.dim, dim)]) / 3;
    this.field[index(0, this.dim + 1, this.dim + 1, dim)] = (this.field[index(1, this.dim + 1, this.dim + 1, dim)] + this.field[index(0, this.dim, this.dim + 1, dim)] + this.field[index(0, this.dim + 1, this.dim, dim)]) / 3;
    this.field[index(this.dim + 1, 0, this.dim + 1, dim)] = (this.field[index(this.dim, 0, this.dim + 1, dim)] + this.field[index(this.dim + 1, 1, this.dim + 1, dim)] + this.field[index(this.dim + 1, 0, this.dim, dim)]) / 3;
    this.field[index(this.dim + 1, this.dim + 1, this.dim + 1, dim)] = (this.field[index(this.dim, this.dim + 1, this.dim + 1, dim)] + this.field[index(this.dim + 1, this.dim, this.dim + 1, dim)] + this.field[index(this.dim + 1, this.dim + 1, this.dim, dim)]) / 3;
};
