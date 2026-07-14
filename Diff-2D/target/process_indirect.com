#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in ./n15_noesy.ft1            \
| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0  -93 -p1  0               \
| nmrPipe  -fn FT    -inv       \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5  \
#| nmrPipe  -fn ZF -size 64                             \
#| nmrPipe  -fn POLY -auto -ord 0                    \
| nmrPipe  -fn ZTP \
#| nmrPipe  -fn LP -fb                               \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5  \
#| nmrPipe  -fn ZF -size 64                             \
#| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0  0 -p1  0                 \
#| nmrPipe  -fn POLY -auto -ord 0                    \
| nmrPipe  -fn ZTP \
| nmrPipe  -fn TP \
  > ./fid_temp.dat

exit
    
