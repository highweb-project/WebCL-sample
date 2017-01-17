function VectorField(dim, viscosity, dt, boundaries) {
    var i, j, k;

    this.field = new Float32Array(getFlatSize(dim) * 3);

    this.dim = dim;
    this.viscosity = viscosity;
    this.dt = dt;
    this.slip = true;
    this.vorticityScale = 3.0;

    for (i = 0; i < this.dim + 2; i++) {
        for (j = 0; j < this.dim + 2; j++) {
            for (k = 0; k < this.dim + 2; k++) {
                this.field[vindex(i, j, k, 0, dim)] = 0.0;
                this.field[vindex(i, j, k, 1, dim)] = 0.0;
                this.field[vindex(i, j, k, 2, dim)] = 0.0;
            }
        }
    }
}

VectorField.prototype.setTimestep = function (value) {
    this.dt = value;
};

VectorField.prototype.setViscosity = function (value) {
    this.viscosity = value;
};

VectorField.prototype.reset = function () {
    var bufSize = 4 * numCells;
    var i, j, k;

    for (i = 0; i < this.dim + 2; i++) {
        for (j = 0; j < this.dim + 2; j++) {
            for (k = 0; k < this.dim + 2; k++) {
                this.field[vindex(i, j, k, 0, dim)] = 0.0;
                this.field[vindex(i, j, k, 1, dim)] = 0.0;
                this.field[vindex(i, j, k, 2, dim)] = 0.0;
            }
        }
    }

    clQueue.enqueueWriteBuffer(vectorBuffer, false, 0, bufSize * 3, this.getField(), []);
};

VectorField.prototype.getField = function () {
    return this.field;
};

VectorField.prototype.draw = function () {

};

VectorField.prototype.step = function (source) {
    this.addField(source);
    this.vorticityConfinement();
    this.diffusion();
    this.projection();
    this.advection();
    this.projection();
};

VectorField.prototype.addField = function (source) {

    vectorAddKernel.setArg(0, vectorBuffer);
    vectorAddKernel.setArg(1, vectorSourceBuffer);
    vectorAddKernel.setArg(2, new Uint32Array([this.dim]));
    vectorAddKernel.setArg(3, new Float32Array([this.dt]));

    try {
        var globalWS = new Int32Array(3);
        globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorAddKernel, globalWS.length, [], globalWS, []);
        clQueue.finish();
        clTime += Date.now() - start;
    } catch (e) {
        console.innerHTML = e;
        console.log("vectorField.addField", [e]);
    }
};

VectorField.prototype.diffusion = function () {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        vectorCopyKernel.setArg(0, vectorBuffer);
        vectorCopyKernel.setArg(1, vectorTempBuffer);
        vectorCopyKernel.setArg(2, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorCopyKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        vectorDiffusionKernel.setArg(0, vectorBuffer);
        vectorDiffusionKernel.setArg(1, vectorTempBuffer);
        vectorDiffusionKernel.setArg(2, new Uint32Array([this.dim]));
        vectorDiffusionKernel.setArg(3, new Float32Array([this.dt]));
        vectorDiffusionKernel.setArg(4, new Float32Array([this.viscosity]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorDiffusionKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("vectorField.diffusion", [e]);
    }

    this.setBoundaryVelocities();
};

VectorField.prototype.projection = function () {
    var globalWS = new Int32Array(3);
    var i;
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        vectorInitFieldKernel.setArg(0, scalarTempBuffer);
        vectorInitFieldKernel.setArg(1, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorInitFieldKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        vectorInitFieldKernel.setArg(0, scalarSecondTempBuffer);
        vectorInitFieldKernel.setArg(1, new Uint32Array([this.dim]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorInitFieldKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        vectorProjectionFirst.setArg(0, vectorBuffer);
        vectorProjectionFirst.setArg(1, scalarTempBuffer);
        vectorProjectionFirst.setArg(2, scalarSecondTempBuffer);
        vectorProjectionFirst.setArg(3, new Uint32Array([this.dim]));
        vectorProjectionFirst.setArg(4, new Float32Array([1 / this.dim]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorProjectionFirst, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        this.setScalarFieldDensities(scalarTempBuffer);
        this.setScalarFieldDensities(scalarSecondTempBuffer);

        for (i = 0; i < 20; i++) {
            vectorProjectionSecond.setArg(0, vectorBuffer);
            vectorProjectionSecond.setArg(1, scalarTempBuffer);
            vectorProjectionSecond.setArg(2, scalarSecondTempBuffer);
            vectorProjectionSecond.setArg(3, new Uint32Array([this.dim]));
            vectorProjectionSecond.setArg(4, new Float32Array([1 / this.dim]));

            start = Date.now();
            clQueue.enqueueNDRangeKernel(vectorProjectionSecond, globalWS.length, [], globalWS, []);
            clTime += Date.now() - start;

            this.setScalarFieldDensities(scalarTempBuffer);
        }

        vectorProjectionThird.setArg(0, vectorBuffer);
        vectorProjectionThird.setArg(1, scalarTempBuffer);
        vectorProjectionThird.setArg(2, scalarSecondTempBuffer);
        vectorProjectionThird.setArg(3, new Uint32Array([this.dim]));
        vectorProjectionThird.setArg(4, new Float32Array([1 / this.dim]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorProjectionThird, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        this.setBoundaryVelocities();
    } catch (e) {
        console.error("vectorField.projection", [e]);
    }
};

VectorField.prototype.advection = function () {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        vectorCopyKernel.setArg(0, vectorBuffer);
        vectorCopyKernel.setArg(1, vectorTempBuffer);
        vectorCopyKernel.setArg(2, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorCopyKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        vectorAdvectionKernel.setArg(0, vectorBuffer);
        vectorAdvectionKernel.setArg(1, vectorTempBuffer);
        vectorAdvectionKernel.setArg(2, new Uint32Array([this.dim]));
        vectorAdvectionKernel.setArg(3, new Float32Array([this.dt]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorAdvectionKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("vectorField.advection", [e]);
    }

    this.setBoundaryVelocities();
};


VectorField.prototype.vorticityConfinement = function () {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        vectorCopyKernel.setArg(0, vectorBuffer);
        vectorCopyKernel.setArg(1, vectorTempBuffer);
        vectorCopyKernel.setArg(2, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorCopyKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        vectorVorticityFirstKernel.setArg(0, vectorBuffer);
        vectorVorticityFirstKernel.setArg(1, vectorTempBuffer);
        vectorVorticityFirstKernel.setArg(2, scalarTempBuffer);
        vectorVorticityFirstKernel.setArg(3, new Uint32Array([this.dim]));
        vectorVorticityFirstKernel.setArg(4, new Float32Array([this.dt * this.vorticityScale]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorVorticityFirstKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;

        vectorVorticitySecondKernel.setArg(0, vectorBuffer);
        vectorVorticitySecondKernel.setArg(1, vectorTempBuffer);
        vectorVorticitySecondKernel.setArg(2, scalarTempBuffer);
        vectorVorticitySecondKernel.setArg(3, new Uint32Array([this.dim]));
        vectorVorticitySecondKernel.setArg(4, new Float32Array([this.dt * this.vorticityScale]));

        start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorVorticitySecondKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("vectorField.vorticityConfinement", [e]);
    }
};

VectorField.prototype.setBoundaryVelocities = function () {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        vectorBoundariesKernel.setArg(0, vectorBuffer);
        vectorBoundariesKernel.setArg(1, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(vectorBoundariesKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("vectorField.setBoundaryVelocities", [e]);
    }
};

VectorField.prototype.setScalarFieldDensities = function (field) {
    var globalWS = new Int32Array(3);
    globalWS[0] = globalWS[1] = globalWS[2] = localThreads;

    try {
        scalarBoundariesKernel.setArg(0, field);
        scalarBoundariesKernel.setArg(1, new Uint32Array([this.dim]));

        var start = Date.now();
        clQueue.enqueueNDRangeKernel(scalarBoundariesKernel, globalWS.length, [], globalWS, []);
        clTime += Date.now() - start;
    } catch (e) {
        console.error("vectorField.setScalarFieldDensities", [e]);
    }
};
