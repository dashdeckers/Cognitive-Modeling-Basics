# Analysis of the data from Acerbi et al. (2012)
# Cognitive Modelling: Basic Principles and Methods 20/21

library(data.table)
library(dplyr)
library(ggplot2)

theme_set(theme_bw(base_size = 14))


# Load and prepare data ---------------------------------------------------
d <- fread("dataAcerbi.csv")

str(d)

# There are some rows with missing data, so let's remove those
d[which(is.na(d)),]
d <- na.omit(d)

# Fix column types
d[, Subject := as.factor(Subject)]

# The presented intervals (int) include noise
# For the analysis, make a discrete version
d[, int_d := mean(int), by = .(Block, stim)]

d[,exp:="Exp1"]
d[grep("Medium",d$Block),exp:="Exp2"]

# Plot data ---------------------------------------------------------------

# Calculate average production times
d_avg <- d[, .(resp_subj = mean(resp)), by = .(Block, int_d, Subject)]
d_avg <- d_avg[, .(resp = mean(resp_subj)), by = .(Block, int_d)]

ggplot(d, aes(x = int_d, y = resp, colour = Block, group = Block)) +
  facet_grid(~ Block) +
  geom_jitter(width = .02, height = 0, alpha = .1, pch=".") +
  geom_line(data = d_avg, lwd = 2) +
  geom_point(data = d_avg, colour = "black", size = 2, pch=".") +
  guides(colour = FALSE) +
  labs(x = "Ts (s)", y = "Tp (s)")

ggplot(d, aes(x=int_d,y=resp,colour=Block)) +
  geom_abline(slope=1, alpha=.5) +
  geom_line(data = d_avg, lwd = 2) +
  geom_point(data = d_avg, colour = "black", size = 2) +
  labs(x = "Ts (s)", y = "Tp (s)") +
  coord_fixed()

## ---

# A difference plot like shown in the paper

if (FALSE) {
  d_avg <- d[exp=="Exp2", .(resp = mean(resp)), by = .(stim, Subject,Block)]
  setorder(d_avg,stim,Subject,Block)

  plotdat <- d_avg[Block=="Medium Peaked",]
  plotdat[,diff := d_avg[Block=="Medium Peaked",resp]-d_avg[Block=="Medium Uniform",resp]]

  plotdat <- plotdat[,mean(diff),by=stim]

  plotdat[,plot(V1)]
}

## ---


library(lme4)
library(lmerTest)

expdat <- d[exp=="Exp2",]
expdat[,intdev := int - mean(c(.600,.975))]
expdat[,cond:=ifelse(Block=="Medium Uniform",.5,-.5)]

summary(lmer(resp ~ intdev*cond + (1 | Subject), data=expdat))
