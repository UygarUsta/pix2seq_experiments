This is a sequence augmentation added version of good_working_pix2seq's quad script. Works well, got higher ap than non sequence augmentation version on "Ruhsat extended dataset".
Got ap of 0.8096 (quad). 

Florence based implementation got ap of 0.8674 on "Ruhsat extended dataset", since it is based on florence implementation it does not use sequence augmentation. Uses task token (even if it is for quadbox only).
