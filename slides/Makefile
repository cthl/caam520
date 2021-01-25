all: *.tex
	latexmk -pdf

clean_tmp:
	rm -f *.out *.toc *.snm *.fls *.fdb_latexmk *.aux *.nav *.log *.synctex.gz *.vrb

clean: clean_tmp
	rm -f *.pdf

.PHONY: all clean_tmp clean
