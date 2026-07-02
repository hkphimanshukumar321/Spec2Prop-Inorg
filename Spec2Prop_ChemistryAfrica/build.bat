@echo off
echo Building main_nonblind...
set BSTINPUTS=bst;
pdflatex -interaction=nonstopmode main_nonblind.tex
bibtex main_nonblind
pdflatex -interaction=nonstopmode main_nonblind.tex
pdflatex -interaction=nonstopmode main_nonblind.tex

echo Building main_blinded...
pdflatex -interaction=nonstopmode main_blinded.tex
bibtex main_blinded
pdflatex -interaction=nonstopmode main_blinded.tex
pdflatex -interaction=nonstopmode main_blinded.tex

echo Build complete!
pause
