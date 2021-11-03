#! /usr/bin/bash
./build/ibpm -ic cylinderLOW/ibpm00500.bin -Re 100 -nx 450 -ny 200 -ngrid 4 -length 9 -xoffset -1 -yoffset -2 -xshift 0.75 -nsteps 50 -restart 50 -tecplot 50 -outdir cylinderLOW/ -geom cylinderLOW/cylinderPLUNGE.geom -dt .002 -ubf 1 -tecplotallgrids 1
