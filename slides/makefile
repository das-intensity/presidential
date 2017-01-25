
F=presidential

.PHONY: pdf ps images graphs

IMGPNG=\

IMGJPG=\

IMGGNU=\

#IMGEPSORIG=$(IMGPNG:.png=.eps) $(IMGJPG:.png=.eps)
#IMGEPS=$(echo "$(IMGPNG)" "$(IMGJPG)" | sed s/\.png/\.eps/g | sed s/baseimages/images/g)
IMGEPS= \
	$(IMGPNG:.png=.eps) $(IMGJPG:.jpg=.eps) $(IMGGNU:.gnu=.eps) \
	# \


default: ps

pdf: $F.pdf
	@:

ps: $F.ps

$F.pdf: $F.ps
	ps2pdf \
		-dSubsetFonts=true \
		-dEmbedAllFonts=true \
		-dMaxSubsetPct=100 \
		-dCompatibilityLevel=1.4 \
		$F.ps
		#-dPDFSETTINGS=/prepress \

continuous: $F.pdf
	evince $F.pdf &
	while :; do make pdf; sleep 2; done

$F.ps: $F.dvi
	dvips $F.dvi \
		#-Pdownload35 \

$F.dvi: $F.tex makefile $(IMGEPS)
	latex -halt-on-error $F
	#bibtex $F
	latex -halt-on-error $F
	latex -halt-on-error $F

show: $F.ps
	evince $F.ps

showpdf: $F.pdf
	evince $F.pdf

images: $(IMGEPS)
	cd .. && make images
	#cd .. && make gnu/len10-t{1,2,3,4}.eps
	
%.eps: %.png
	convert $< `echo $@ | sed s/\.png/\.eps/g`

%.eps: %.jpg
	convert $< `echo $@ | sed s/\.jpg/\.eps/g`
	
gnu: $(IMGGNU:.gnu=.eps)

%.eps: %.gnu
	cd gnu && gnuplot < `echo $< | sed 's/gnu\///g'`

clean:
	rm -rf $F.{ps,dvi,aux,log,bbl,blg,out,nav,snm,toc} \
	texput.log \

purge: clean
	rm -f $F.pdf
	for i in $(IMGPNG); do rm -f `echo $$i | sed s/\.png/\.eps/g`; done
	for i in $(IMGJPG); do rm -f `echo $$i | sed s/\.jpg/\.eps/g`; done
	rm -f $(IMGGNU:.gnu=.eps)

