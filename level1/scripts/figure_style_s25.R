suppressPackageStartupMessages({library(ggplot2);library(grid);library(patchwork);library(jsonlite);library(ragg)})
ff <- "Arial"; ink <- "#20252B"; muted <- "#667085"; b0 <- "#A5ADB6"; b1 <- "#5E6873"
plm <- "#009E73"; blue <- "#0072B2"; amber <- "#A66500"; red <- "#B2472B"
th25 <- function() theme_classic(base_size=8.5,base_family=ff)+theme(
 text=element_text(colour=ink),axis.text=element_text(size=8.2,colour=ink),
 axis.title=element_text(size=8.5),axis.line=element_line(linewidth=.35),
 axis.ticks=element_line(linewidth=.3),legend.position="bottom",legend.title=element_blank(),
 legend.text=element_text(size=8.2),legend.key.width=unit(10,"pt"),
 plot.tag=element_text(face="bold",size=10),plot.tag.position=c(0,1),
 plot.margin=margin(6,8,6,6),strip.background=element_blank(),strip.text=element_text(size=8.5,face="bold"))
save25 <- function(fig,out,name,w,h,dpi=600) {
 dir.create(out,recursive=TRUE,showWarnings=FALSE)
 ggsave(file.path(out,paste0(name,".pdf")),fig,width=w,height=h,units="mm",device=cairo_pdf,family=ff,bg="white")
 agg_tiff(file.path(out,paste0(name,".tiff")),width=w,height=h,units="mm",res=dpi,compression="lzw",background="white");print(fig);dev.off()
 agg_png(file.path(out,paste0(name,".png")),width=w,height=h,units="mm",res=300,background="white");print(fig);dev.off()
}
