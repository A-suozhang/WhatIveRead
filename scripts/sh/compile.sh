FILE=${1:-"ms.tex"}
echo $FILE
# ${FILE/.tex/.aux}


xelatex ${FILE}
bibtex ${FILE/.tex/.aux}
xelatex ${FILE}
pdflatex ${FILE}

rm *.bbl
rm *.aux
rm *.log
rm *.blg


