# Capacity and response-distillation effects are separate, paired comparisons.
sp<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)));pkg<-dirname(dirname(sp))
source(file.path(pkg,'scripts/figure_style_s25.R'))
r<-fromJSON(file.path(pkg,'analysis_addon/results/distillation_capacity_20260909.json'),simplifyVector=FALSE)
s<-r$models$semantic;li<-r$models$linear
labels<-c('B1','Context\ncontrol','Linear\ndistill.','Small pLM\nhard-only','Small pLM\ndistilled','ProtT5\nteacher')
vals<-c(s$metrics$b1$auc,r$transparent_control$metrics$selected$auc,li$metrics$distilled$auc,s$metrics$hard_tuned$auc,s$metrics$distilled$auc,s$metrics$teacher$auc)
d<-data.frame(model=factor(labels,levels=labels),auc=vals,colour=c(b1,b0,blue,b0,plm,plm),shape=c(15,1,16,1,16,17))
pa<-ggplot(d,aes(model,auc))+geom_hline(yintercept=s$metrics$b1$auc,linetype='dashed',colour=muted,linewidth=.35)+
 geom_point(aes(colour=colour,shape=shape),size=2.8)+scale_colour_identity()+scale_shape_identity()+
 geom_text(aes(label=sprintf('%.3f',auc)),nudge_y=.018,family=ff,size=2.9)+
 scale_y_continuous(limits=c(.5,.83),breaks=c(.5,.6,.7,.8),expand=c(0,0))+
 labs(x=NULL,y='ROC-AUC',tag='a')+th25()+theme(legend.position='none')
contrasts<-list(r$transparent_control$distilled_minus_selected_transparent,r$models$rbf$distilled_minus$prior_transparent,r$models$neural$distilled_minus$transparent_blend,s$distilled_minus$hard_tuned)
names<-c('Linear / context','RBF / context','Neural / context','Small pLM / hard')
ef<-data.frame(compare=factor(names,levels=rev(names)),delta=vapply(contrasts,`[[`,numeric(1),'observed_delta'),lo=vapply(contrasts,function(x)x$ci95[[1]],numeric(1)),hi=vapply(contrasts,function(x)x$ci95[[2]],numeric(1)))
pb<-ggplot(ef,aes(delta,compare))+geom_vline(xintercept=0,linetype='dashed',colour=muted,linewidth=.35)+
 geom_segment(aes(x=lo,xend=hi,yend=compare),linewidth=.9,colour=plm)+geom_point(size=2.5,colour=plm)+
 geom_text(aes(label=sprintf('%+.4f [%.4f, %.4f]',delta,lo,hi)),nudge_y=.29,family=ff,size=2.65)+
 scale_x_continuous(limits=c(-.012,.021),breaks=c(-.01,0,.01,.02),labels=c('−.01','0','+.01','+.02'),expand=c(0,0))+
 scale_y_discrete(expand=expansion(add=.6))+labs(x='Distilled − stronger hard control AUC',y=NULL,tag='b')+th25()+
 theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),axis.text.y=element_text(size=7.8))
e<-r$semantic_engineering;l<-r$linear_engineering
stopifnot(e$full_parameters==33762666,l$inference_coefficients==2925)
# Booktabs-style typography: sparse horizontal rules, aligned numeric columns.
rows<-c('Parameters','Weights','CPU time')
left<-c('2,925',sprintf('%.1f kB',l$source_npz_bytes/1000),sprintf('%.3f ms',l$timings[['500']]$median_ms))
teacher<-c('≈1.2 B','≈4.8 GB*','—')
right<-c('33.76 M',sprintf('%.1f MB',e$full_weight_artifact_bytes/1e6),sprintf('%.0f ms',e$timings[['500']]$sequence_to_score_median_ms))
pc<-wrap_elements(full=grobTree(
 textGrob('c',x=.01,y=.98,just=c('left','top'),gp=gpar(fontfamily=ff,fontsize=10,fontface='bold')),
 segmentsGrob(x0=.06,x1=.96,y0=c(.83,.67,.21),y1=c(.83,.67,.21),gp=gpar(col=ink,lwd=c(.9,.45,.9))),
 textGrob(c('Linear','Small pLM','ProtT5'),x=c(.45,.70,.97),y=.75,just='right',gp=gpar(fontfamily=ff,fontsize=8.6,fontface='bold')),
 textGrob(rows,x=.03,y=c(.56,.43,.30),just='left',gp=gpar(fontfamily=ff,fontsize=8.5)),
 textGrob(left,x=.45,y=c(.56,.43,.30),just='right',gp=gpar(fontfamily=ff,fontsize=8.5)),
 textGrob(right,x=.70,y=c(.56,.43,.30),just='right',gp=gpar(fontfamily=ff,fontsize=8.5)),
 textGrob(teacher,x=.97,y=c(.56,.43,.30),just='right',gp=gpar(fontfamily=ff,fontsize=8.5)),
 textGrob('*FP32 parameter estimate',x=.97,y=.10,just='right',gp=gpar(fontfamily=ff,fontsize=7.5,col=muted))))
# Two visual routes: training supplies a head; inference retains the small encoder.
flow<-grobTree(textGrob('d',x=.01,y=.98,just=c('left','top'),gp=gpar(fontfamily=ff,fontsize=10,fontface='bold')))
node<-function(x,w,label,colour,fill)grobTree(
 roundrectGrob(x=x,y=.38,width=w,height=.42,r=unit(1.5,'mm'),gp=gpar(fill=fill,col=colour,lwd=.7)),
 textGrob(label,x=x,y=.38,gp=gpar(fontfamily=ff,fontsize=9,col=colour,fontface='bold')))
flow<-addGrob(flow,grobTree(node(.16,.21,'Sequence',ink,'white'),node(.45,.27,'Small pLM + B1',blue,'#EFF6FA'),node(.75,.18,'Head',plm,'#EDF8F3')))
for(pair in list(c(.27,.31),c(.59,.65),c(.85,.91)))flow<-addGrob(flow,segmentsGrob(x0=pair[1],x1=pair[2],y0=.38,y1=.38,arrow=arrow(length=unit(1.5,'mm'),type='closed'),gp=gpar(col=ink,lwd=.9)))
flow<-addGrob(flow,grobTree(textGrob('Score',x=.955,y=.38,gp=gpar(fontfamily=ff,fontsize=9)),
 textGrob('Teacher + labels',x=.75,y=.89,gp=gpar(fontfamily=ff,fontsize=8.5,col=plm)),
 segmentsGrob(x0=.75,x1=.75,y0=.76,y1=.61,arrow=arrow(length=unit(1.5,'mm'),type='closed'),gp=gpar(col=plm,lwd=.9,lty='dashed'))))
pd<-wrap_elements(full=flow)
fig<-(pa/(pb|pc)/pd)+plot_layout(heights=c(1,.98,.40))+plot_annotation(theme=theme(plot.margin=margin(4,4,4,4),plot.background=element_rect(fill='white',colour=NA)))
save25(fig,file.path(pkg,'figures'),'Okada_Fig5',172,152,dpi=1200)
