suppressPackageStartupMessages({library(grid); library(ragg)})
args <- grep('^--file=', commandArgs(FALSE), value=TRUE)
root <- normalizePath(file.path(dirname(sub('^--file=', '', args[1])), '..'))
out <- file.path(root, 'figures')
ink <- '#243746'; blue <- '#0072B2'; green <- '#009E73'; gray <- '#6B7883'
label <- function(s,x,y,size=8,face='plain',col=ink) grid.text(s,unit(x,'mm'),unit(y,'mm'),gp=gpar(fontfamily='Arial',fontsize=size,fontface=face,col=col,lineheight=1.15))
node <- function(x,y,title,detail,col,fill='white') {
 grid.roundrect(unit(x,'mm'),unit(y,'mm'),unit(73,'mm'),unit(17,'mm'),r=unit(1.4,'mm'),gp=gpar(col=col,fill=fill,lwd=.7))
 label(title,x,y+3,10.5,'bold');label(detail,x,y-3.5,8.5)
}
arrow <- function(x,y0,y1) grid.segments(unit(x,'mm'),unit(y0,'mm'),unit(x,'mm'),unit(y1,'mm'),arrow=grid::arrow(type='closed',length=unit(1.4,'mm')),gp=gpar(col=gray,lwd=.8))
draw <- function(){
 grid.newpage();grid.rect(gp=gpar(fill='white',col=NA))
 label('a',5,191,11,'bold');label('b',91,191,11,'bold')
 ys <- c(176,152,128,104,80,56)
 left <- list(c('Define the prediction task','Endpoint, chemistry and conditions'),c('Select compatible records','Deduplicate; keep independent groups'),c('Separate training and test groups','Choose splits for intended use'),c('Fit and tune in training groups','B0 → B1 → B1 + pLM'),c('Evaluate untouched groups','Paired gain, calibration, applicability'),c('Validate in matched experiments','Fix the model and decision rule'))
 right <- list(c('Start with a useful teacher','Fix the student and deployment budget'),c('Cross-fit teacher probabilities','Exclude each target row’s group'),c('Train two matched students','Hard labels only  |  Labels + teacher'),c('Tune within training groups','Independent tuning for both students'),c('Evaluate untouched groups','Distillation gain and teacher retention'),c('Profile the complete predictor','Encoder + head: size, time, memory'))
 for(i in seq_along(ys)){
  node(43,ys[i],left[[i]][1],left[[i]][2],blue,if(i==5)'#EDF6FB' else 'white')
  node(129,ys[i],right[[i]][1],right[[i]][2],green,if(i==5)'#EDF9F5' else 'white')
  if(i<6){arrow(43,ys[i]-9,ys[i+1]+9);arrow(129,ys[i]-9,ys[i+1]+9)}
 }
 arrow(129,47,38)
 node(129,28,'Export only if criteria are met','After redesign, obtain a fresh test',green,'#EDF9F5')
 label('Compare predictive value',43,27,10.5,'bold',blue)
 label('before investing in compression',43,21,8.5)
}
cairo_pdf(file.path(out,'Okada_Fig2.pdf'),width=172/25.4,height=198/25.4,family='Arial');draw();dev.off()
agg_png(file.path(out,'Okada_Fig2.png'),width=172,height=198,units='mm',res=300);draw();dev.off()
agg_tiff(file.path(out,'Okada_Fig2.tiff'),width=172,height=198,units='mm',res=600,compression='lzw');draw();dev.off()
