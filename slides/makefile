
F=presidential

.PHONY: pdf ps images graphs

IMGPNG=\
	   images/face-tracking.png \
	   images/fpv-hud.png \
	   images/hsv.png \
	   images/hyperspectral.png \
	   images/logo_dark_transparent.png \
	   images/logo_final.png \
	   images/mantis-shrimp-blue.png \
	   images/mantis-shrimp-green.png \
	   images/mantis-shrimp-grey.png \
	   images/mantis-shrimp-red.png \
	   images/numpy.png \
	   images/squirrel.png \
	   images/step-squirrels.png \
	   images/step-webcam.png \
	   images/step-squirrel-productions.png \
	   images/step-face-detection.png \
	   images/step-presidential-skin.png \
	   images/step-presidential-hair.png \
	   images/tmv-logo.png \
	   images/tmv-results.png \

IMGJPG=\
	   images/camera-transformer.jpg \
	   images/computer-eye.jpg \
	   images/mag-hog.jpg \
	   images/mantis-shrimp.jpg \
	   images/nasa-robot.jpg \
	   images/tmv-logo-hog.jpg \

IMGSVG=\
	   images/mag.svg \
	   images/opencv.svg \
	   images/python.svg \

IMGGNU=\

#IMGEPSORIG=$(IMGPNG:.png=.eps) $(IMGJPG:.png=.eps)
#IMGEPS=$(echo "$(IMGPNG)" "$(IMGJPG)" | sed s/\.png/\.eps/g | sed s/baseimages/images/g)
IMGEPS= \
	$(IMGPNG:.png=.eps) $(IMGJPG:.jpg=.eps) $(IMGGNU:.gnu=.eps) $(IMGSVG:.svg=.eps) \
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

%.eps: %.svg
	convert -trim $< `echo $@ | sed s/\.svg/\.eps/g`
	
gnu: $(IMGGNU:.gnu=.eps)

%.eps: %.gnu
	cd gnu && gnuplot < `echo $< | sed 's/gnu\///g'`

clean:
	rm -rf $F.{ps,dvi,aux,log,bbl,blg,out,nav,snm,toc,vrb} \
	texput.log \

purge: clean
	rm -f $F.pdf
	rm -f $(IMGJPG:.jpg=.eps)
	rm -f $(IMGPNG:.png=.eps)
	rm -f $(IMGGNU:.gnu=.eps)
	rm -f $(IMGSVG:.svg=.eps)

