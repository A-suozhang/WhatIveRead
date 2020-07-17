FILE=${1:-"ms.tex"}
echo $FILE
# ${FILE/.tex/.aux}


xelatex ${FILE}
bibtex ${FILE/.tex/.aux}
xelatex ${FILE}
pdflatex ${FILE}


