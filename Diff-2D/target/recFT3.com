#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimension
proj3D.tcl -in ./fidtemp3D.ft3 -abs

nmrPipe -in 15N.13C.dat                          \
| nmrPipe  -ov -out  ./labeltempCN.dat

nmrDraw labeltempCN.dat
