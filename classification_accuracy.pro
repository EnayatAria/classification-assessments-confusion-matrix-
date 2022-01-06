PRO classification_accuracy
  ENVI, /restore_base_save_files
  ENVI_BATCH_INIT, LOG_FILE = 'batch.log'
  ;image_file = ENVI_PICKFILE(TITLE = 'select an Image')
  ;IF (image_file EQ "") THEN RETURN
 ; for i=2,32 do begin
  ;Open the input file after removing the noisy channels
  
  envi_open_file, $
    'C:\Users\enaya\Desktop\article-13-7-2019\International Journal ohf RS\implementation\Salinas\190_bands_PCA',$
   ; 'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\IndPin_178b_Qc_P',$
    ; 'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\Salinas\190_bands', $
    r_fid=fid
  if (fid eq -1) then begin
    envi_batch_exit
    return
  endif
  envi_file_query,fid, dims=dims, nb=nb1, wl=wl
  pos_gen  = lindgen(nb1)
  envi_restore_rois, $
    'C:\Users\enaya\Desktop\article-13-7-2019\International Journal ohf RS\implementation\Salinas\16_small_cls.roi'
     ;'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\13classes.roi'
    ;'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\13clas_bigTR.roi
    ;'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\Salinas\16_classes.roi'
    ;'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\Salinas\16_small_cls.roi'
  roi_ids = envi_get_roi_ids(fid=fid,roi_colors=roi_colors, roi_names=class_names)
  class_names = ['Unclassified', class_names]
  class_names = [class_names]
  num_classes = n_elements(roi_ids)
  lookup = bytarr(3,num_classes+1)
  lookup[0,1] = roi_colors
  ENVI_GET_ROI_INFORMATION, roi_ids, npts=npts, nl=nl, ns=ns
  ; make final image and then classify it
  ;******************************************************************
 
;  data=intarr(dims(2)+1,dims(4)+1,nb1)
;  for i=0, nb1-1 do begin
;    a=envi_get_data(dims=dims,fid=fid,pos=pos_gen(i))
;    neg=where(a le 0,c)
;    if c gt 0 then a(neg)=1
;    data(*,*,i) = temporary(a)
;  endfor
;  data=double(data)
;  if pp le 9 then $
;         envi_open_file, $
;           'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\Bandselection\salinas\SRS_JM_bands\' $
;               +string(pp, format='(I1)')+'b', r_fid=fid $
    ;  else  envi_open_file, $
    ;        'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\Bandselection\salinas\SRS_JM_bands\' $
    ;              +string(pp, format='(I2)')+'b', r_fid=fid
    ;  if (fid eq -1) then begin
    ;    envi_batch_exit
    ;    return
    ;  endif
;    result=dblarr(5,21)
;    openr, lun, $
;      ;'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\spectral representation\IEEE\splitting_rec_result_Indian_SID-SA.txt',/get_lun
;    readf,lun,result
;    free_lun, lun
 ; for pp=2,20 do begin  
    
;    ;**********************************************************************
;    ;Reading the files of different bandsets creadetd by spectral region merging 
;    ; and then combining them to make the final images 
;   ;openr, lun, $
; ;   'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\information content\classification\Indian Pines\channel selection\STM-TD\Based on correlation\Final_split_Result_SRM_'+string(pp, format='(I2)')+'.txt',/get_lun
;  ;readf,lun,result
;  ;free_lun, lun
;  
;; ;final_res=result(0:pp-1,pp-1)
;  final_res=result(1,0:pp-2)
;; ; s=size(final_res)
;;;  ;final_res=final_res(0:s(2)-2)
;;;  fid_arr=lonarr(200)
;;;  ;arrange=final_res(0:i)
;    arrange=final_res(sort(final_res))
;    ss=size(arrange)
;    data1=fltarr(dims(2)+1,dims(4)+1,ss(1)+1)
;   for j=0, ss(1)-1 do begin
;    if j ne 0  then begin
;        while arrange(j) eq arrange(j-1) do begin
;          data1(*,*,j)=data1(*,*,j-1)
;          j=j+1
;          if  j eq ss(1) then break
;        endwhile
;      endif
;      if  j eq ss(1) then break
;      
;      if j eq 0 then begin
;        L1=0
;        L2=arrange(j)-1
;      endif else  begin
;        L1=arrange(j-1)
;        L2=arrange(j)-1
;      endelse
;      if l1 eq l2 then data1(*,*,j)=data(*,*,L1:L2) else  data1(*,*,j)=mean(data(*,*,L1:L2), dimension=3)
;       if j eq ss(1)-1 then begin
;        L1=arrange(j)
;        L2=nb1-1
;        if l1 eq l2 then data1(*,*,j+1)=data(*,*,L1:L2) else  data1(*,*,j+1)=mean(data(*,*,L1:L2), dimension=3)
;      endif
;    endfor
;
;   data2=data1 
;    envi_enter_data, data1,r_fid=fid1
;    
;    envi_file_query,fid1, dims=dims, nb=nb
;    pos_gen  = lindgen(nb)
;  
;  ;***********************************************************************
  
  ; for pp=2,29 do begin 
 
;  if pp le 9 then $
;     envi_open_file, $
;       'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\Bandselection\salinas\SRS_JM_bands\' $
;           +string(pp, format='(I1)')+'b', r_fid=fid $ 
;  else  envi_open_file, $
;        'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\Bandselection\salinas\SRS_JM_bands\' $
;              +string(pp, format='(I2)')+'b', r_fid=fid 
;  if (fid eq -1) then begin
;    envi_batch_exit
;    return
;  endif
  

  ; Set the unclassified class to black and use roi colors
  
  
  ;for i=2,91 do begin
  
;;
  ; reading the selected channel from a file 
 ; result=dblarr(190)
; result=dblarr(2,30)
;openr, lun, $
; 'C:\Users\enaya\Desktop\article-13-7-2019\International Journal ohf RS\implementation\ICA_based Band selection_Salinas.txt', /get_lun
;  ;'C:\Users\enaya\Desktop\article-13-7-2019\International Journal ohf RS\implementation\ICA_based Band selection_Indian_pine.txt', /get_lun
; ;   'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\SPIE submission\SVM-RFE\ftRank_Salinas.txt',/get_lun
;readf,lun,result
;free_lun, lun
;;splits=result(2:28,0)-1


;result=dblarr(5,21)
 ; openr, lun, $
; 'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\spectral representation\IEEE\splitting_rec_result_Salinas_SID-SA.txt',/get_lun
;readf,lun,result
;free_lun, lun
;splits=result(2:28,0)-1
;
;  endfor
 ; pos_g=[114,73,30,177,72,1,31,117,74,2,71,28,75,178,56,12,85,32,3,84,4,57,94,55,29,58,93,54,13,59]-1; Indian Pines TD=total dependence
 ; pos_g=[25,56,101,124,14,102,104,143,103,88,152,86,129,168,133,149,44,164,153,16,132,30,173,19,155,93,64,6,70,126]-1; Indian Pines MEAC
 ; pos_g=[11,53,86,127,75,142,34,169,77,62,49,35,106,71,76,16,6,32,27,140,29,31,42,58,55,2,23,72,129,155,119]-1 ; salinas JM SFS
 ;pos_g=[11,49,62,75,129,34,57,90,77,106,32,27,16,6,55,70,53,76,35,47,2,119,29,31,155,145,143,74,59,44,92]-1; Salinas SFS TD
 ; pos_g=[25,56,115,145,33,10,27,132,29,68,20,36,89,102,128,16,158,73,28,127,133,12,24,143,154,30,32,19,53,88,46]-1; SFS_BTS_JM
 ; pos_g=[24,56,116,144,33,10,29,132,27,89,159,16,102,36,128,66,30,20,115,25,73,133,145,12,88,28,21,127,38,100,152]-1; SFS_BTS_TD
  ; pos_g=[115,145,64,33,10,27,25,132,29,89,20,56,36,102,128,158,16]-1 ; Indian Pines JM SFFS 
 ;  pos_g=[147,119,64,29,36,12,132,27,25,16,159,32,89,20,128,100,106,28,133,148,73,2,115,30,13,19,24,53,127,79,35]-1;Indian Pines TD SFFS
 ; pos_g=[49,86,113,75,140,16,29,31,6,34,76,62,166,55,70,102,129]-1; Salinas JM SFFS
; pos_g=[49,129,29,75,35,31,62,55,70,76,106,16,6,90,145,58,50,74,119,34,155,53,77,27,143,2,72,61,44,91,10] ; Salinas TD SFFS

  for pp=1, 21 do begin
;  ;  pp=2
;        ;pos_gen=final_res-1
;  ; pos_gen=lindgen(pp)
    nb=pp+1
  ;pos_gen=result(1,0:pp-1)
; ; pos_gen=result(0:pp-1,pp-1)-1
pos_gen=[0:pp]
    data1=intarr(dims(2)+1,dims(4)+1,nb)
    for i=0, nb-1 do begin
      data1(*,*,i) = envi_get_data(dims=dims, fid=fid, pos=pos_gen(i))
    endfor
    data2=data1
        envi_enter_data, data1,r_fid=fid1
         envi_file_query,fid1, dims=dims, nb=nb
        pos_gen  = lindgen(nb)
    avg = dblarr(nb, num_classes)
;    stdv=fltarr(nb, num_classes)
    cov=dblarr(nb,nb,num_classes)
;    
;    ;stdv=stdv+3
    for j=0, num_classes-1 do begin
;      ; get the statistics for each selected class
      roi_dims=[envi_get_roi_dims_ptr(roi_ids[j]),0,0,0,0]
        a=envi_get_roi_data(roi_ids[j], fid=fid1, pos=pos_gen)
         si=size(a) 
;         negzer=where(a le 0,c)
;;         if c gt 0 then begin
;;          row_arr=fltarr(c)
;;          for i=0, c-1 do begin 
;;            ncol = si(1)
;;            col = negzer(i) MOD ncol
;;            row = negzer(i)/ ncol
;;           row_arr(i)=row
;;          endfor
;;           a=removerows(a,row_arr)
;;           si=size(a)        
;;         endif
;;     
;       ; 60% of data chosen randomly as the training site for classification  
;      ; c=si(2)*60/100
;      ;  b=a(*,round(c*randomu(seed, c)))
;       ; half of the data decussuately fro training
;    ;   b=temporary(a(*,0:si(2)-1:2))
  b=a
        av=mean(b, dimension=2)
;        st=stddev(b, dimension=2)
        cv=correlate(b,/covariance)
;;       ; surface, cv
;     ;   print, av,st
;;      envi_doit, 'envi_stats_doit', fid=fid, pos=pos_gen, $
;;        dims=roi_dims, comp_flag=4, mean=c_mean ,stdv=c_stdv, cov=c_cov
 ;     avg[0,j] = c_mean
;;      stdv[0,j] = c_stdv
;      cov(*,*,j)=c_cov
       avg[0,j] = temporary(av)
;        stdv[0,j] = temporary(st)
        cov(*,*,j)=temporary(cv)
    endfor
;    
    ; cls_img=MLC(data2, avg, cov)
   ; envi_enter_data, cls_img,r_fid=r_fid
;   
;   ;  envi_enter_data, cls_img,r_fid=r_fid, CLASS_NAMES=class_names, LOOKUP=lookup, NUM_CLASSES=17
;   ;out_name='D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\Classification\big_trainingsites\50%_SFFS_TD_MLC_BTS\output' $
;   ; +string(nb)+'b'
;    ; STD_MULT is specifying the width around the standard deviation
   ; ENVI_DOIT, 'CLASS_DOIT', CLASS_NAMES=class_names,$
   ;   DIMS=dims, FID=fid1,  LOOKUP=lookup,    METHOD=3, POS=pos_gen,   $
   ;   r_fid=r_fid, mean=avg, /IN_MEMORY, cov=cov, npts=npts, thresh=0.05; stdv=stdv,data_scale=1.0,, out_name=out_name  
;    
;    
;     svm classifier
  ENVI_DOIT, 'ENVI_SVM_DOIT', DIMS=dims, FID=fid1  , KERNEL_TYPE=2 ,/IN_MEMORY $
    ,  POS=pos_gen,R_FID=r_fid, ROI_IDS=roi_ids, penalty=120, thresh=0.05 ;, OUT_NAME=out_name
;  
    ; reading ground truth image
    envi_open_file, $
      'C:\Users\enaya\Desktop\article-13-7-2019\International Journal ohf RS\implementation\Salinas\reference_map',$
     ;'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\ref_13class',$
     ; 'D:\shosseiniaria\Enayat documents\PhD Thesis\chapters\separability\data\Salinas\reference_map', $
      r_fid=gtfid
    if (gtfid eq -1) then begin
      envi_batch_exit
      return
    endif
    envi_file_query, gtfid, dims=gtdims, nb=nb_gt, num_classes=num_classes_gt, class_names=gt_names
    gtpos=0
    
    class_ptr = lindgen(n_elements(roi_ids)+1)
    ;class_ptr = lindgen(3)
  ; gt_ptr=[0,8,13,1,9,11,10,2,12,7,3,4,5,6]; corresponding to 13classes.roi
  
   ; gt_ptr=[0,1,2,3,4,5,6,7,8,9,10,11,12,13] ; corresponding to 13clas_bigTR.roi
     gt_ptr=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] ; salinas
    pos=0
    ; ENVI_DOIT, 'CLASS_CONFUSION_DOIT',accuracy=accuracy, CALC_PERCENT=1, CFID=r_fid,class_ptr=class_ptr, $
    ; cpos=pos, dims=dims , roi_ids=roi_ids,/rpt_commission, /to_screen, commission=commission, $
    ;  omission=omission, matrix=matrix, kappa_coeff=kappa_coeff, gt_names=class_names
    ; gt_ptr(1:13), class_ptr=class_ptr(1:13), and GT_NAMES=class_names(1:13) are for
    ; removing unclassified class from the classification result 
    ENVI_DOIT, 'CLASS_CONFUSION_DOIT',accuracy=accuracy, CALC_PERCENT=2, CFID=r_fid, class_ptr=class_ptr(1:n_elements(roi_ids)), $
      gt_ptr=gt_ptr(1:n_elements(roi_ids)), cpos=pos, GT_NAMES=class_names(1:n_elements(roi_ids)), dims=dims , GTFID=gtfid, GTPOS=gtpos, GTDIMS=gtdims ;$
    ; , /rpt_commission, /to_screen, commission=commission,  $
    ;  omission=omission, matrix=matrix, kappa_coeff=kappa_coeff
    print, pp+1,  accuracy
;  endfor
;a=transpose(gt_names)
;b=transpose(gt_names(gt_ptr))
;c=transpose(class_names(class_ptr))
;  for jj=0,13 do begin
;    print, jj,"-", b(0,jj), "    ",jj,"-", c(jj)
;  endfor
  endfor
  
  ;    envi_batch_exit
end
