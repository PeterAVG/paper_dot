# setwd("~/.")
print(getwd())
library(ctsmr)
library(splines)
library(MASS)

BASE_FOLDER <- "tex/figures/"

# My learnins with CTSMR:
# don't restrict variables too much, but sometimes it is also good to do.
# start guesses are really important for convergence.
# Every time we add an input, it takes way more time unless we have good start guesses
# There is great multiplicity in solutions. Also, it really helps to smoothen inputs to the model as well.

## read data
X <- read.csv("data/dot_nordic_sde.csv", sep = ",", header = TRUE)
X <- subset(X, select = c(t, Temp_Upp_Dec, Temp_Down_Dec, KW_UZ, KW_LZ, regime_UZ))
names(X) <- c("t", "yTwu", "yTwl", "PtUp", "PtLow", "reg")
XCopy <- X

lb <- 6400
ub <- 7800
X <- X[lb:ub, ]
print(dim(X))

Ta <- 20 # ambient temperature in factory
dt_ <- 1 / 60
setpoint_u <- 455
setpoint_l <- 447

model <- ctsm()
# model$addSystem(dTzu ~ 1 / Czu * (1 / Rzuzl * (Tzl - Tzu) + 1 / Rwz * (Twu - Tzu) + reg / Rza * (20 - Tzu)) * dt + exp(e1) * dw1)
model$addSystem(dTzu ~ 1 / Czu * (1 / Rzuzl * (Tzl - Tzu) + 1 / Rwz * (Twu - Tzu)) * dt + exp(e1) * dw1)
model$addSystem(dTzl ~ 1 / Czl * (1 / Rzuzl * (Tzu - Tzl) + 1 / Rwz * (Twl - Tzl)) * dt + exp(e2) * dw2)

model$addSystem(dTwu ~ 1 / Cwu * ((1 - reg) * 1 / Rwua1 * (20 - Twu) + 1 / Rww * (Twl - Twu) + 1 / Rwz * (Tzu - Twu) + reg / Rwua2 * (20 - Twu) + PtUp) * dt + exp(e3) * dw3)
model$addSystem(dTwl ~ 1 / Cwl * (1 / Rwla1 * (20 - Twl) + 1 / Rww * (Twu - Twl) + 1 / Rwz * (Tzl - Twl) + PtLow) * dt + exp(e4) * dw4)

# X$Pt2 = X$Pt
model$addInput(PtLow, PtUp, reg)
model$addObs(yTwu ~ Twu)
model$addObs(yTwl ~ Twl)
model$setVariance(yTwu ~ exp(e5))
model$setVariance(yTwl ~ exp(e6))

# BEST PARAMETERS FOR 4th order model
# Tzu0    Tzl0    Twu0    Twl0     Cwl     Cwu     Czl     Czu      e1      e2      e3      e4      e5      e6   Rwla1   Rwua1   Rwua2     Rww     Rwz   Rzuzl 
# 455.000 447.000 445.800 445.800   5.997   5.998   1.029   0.502   4.578   3.128  -0.474  -1.110 -21.008  -7.961  11.743   3.795   2.581   0.829   0.185  48.000 

# model$setParameter(Rwua1 = c(init = 3.795, lb = 2, ub = 20))
model$setParameter(Rwua1 = c(init = 3.795))
# model$setParameter(Rwua1 = c(init = fit$xm[["Rwua1"]]))
# model$setParameter(Rwua2 = c(init = 2.581, lb = 2, ub = 20))
model$setParameter(Rwua2 = c(init = 2.581))
# model$setParameter(Rwua2 = c(init = fit$xm[["Rwua2"]]))
# model$setParameter(Rwla1 = c(init = 11.743, lb = 0.1, ub = 20))
model$setParameter(Rwla1 = c(init = 11.743))
# model$setParameter(Rwla1 = c(init = fit$xm[["Rwla1"]]))
# model$setParameter( Rwla2 = c(init=2.849,lb=0.1 ,ub=20 ))
# model$setParameter(Rwla2 = c(init = 4))
# model$setParameter(Rwla2 = c(init = fit$xm[["Rwla2"]]))

# model$setParameter(Rzuzl = c(init = 40, lb = 0.001, ub = 50))
model$setParameter(Rzuzl = c(init = 48))
# model$setParameter(Rzuzl = c(init = fit$xm[["Rzuzl"]]))
model$setParameter(Czl = c(init = 5, lb = 0.5, ub = 20))
# model$setParameter(Czl = c(init = 0.961))
# model$setParameter(Czl = c(init = fit$xm[["Czl"]]))
model$setParameter(Czu = c(init = 5, lb = 0.5, ub = 20))
# model$setParameter(Czu = c(init = 0.595))
# model$setParameter(Czu = c(init = fit$xm[["Czu"]]))

# model$setParameter(Rza = c(init = 5.5, lb = 5, ub = 50))
# model$setParameter(Rza = c(init = 0.5))
# model$setParameter(Rza = c(init = fit$xm[["Rza"]]))
model$setParameter(Rwz = c(init = 0.1, lb = 0.001, ub = 2))
# model$setParameter(Rwz = c(init = 0.2))
# model$setParameter(Rwz = c(init = fit$xm[["Rwz"]]))
model$setParameter(Rww = c(init = 0.2, lb = 0.001, ub = 15))
# model$setParameter(Rww = c(init = 13.767))
# model$setParameter(Rww = c(init = fit$xm[["Rww"]]))
model$setParameter(Cwu = c(init = 3.999, lb = 0.1, ub = 6))
# model$setParameter(Cwu = c(init = 3.999))
# model$setParameter(Cwu = c(init = fit$xm[["Cwu"]]))
model$setParameter(Cwl = c(init = 3.998, lb = 0.1, ub = 6))
# model$setParameter(Cwl = c(init = 3.998))
# model$setParameter(Cwl = c(init = fit$xm[["Cwl"]]))

model$setParameter(e1 = c(init = 4, lb = -50, ub = 5))
model$setParameter(e2 = c(init = 2.7, lb = -50, ub = 5))
model$setParameter(e3 = c(init = -1.2, lb = -50, ub = 5))
model$setParameter(e4 = c(init = -3, lb = -50, ub = 5))
model$setParameter(e5 = c(init = -20, lb = -50, ub = 5))
model$setParameter(e6 = c(init = -7, lb = -50, ub = 5))
model$setParameter(Tzu = c(init = setpoint_u))
model$setParameter(Tzl = c(init = setpoint_l))
model$setParameter(Twu = c(init = X$yTwl[1]))
model$setParameter(Twl = c(init = X$yTwl[1]))

fit <- model$estimate(X, firstorder = TRUE, threads = 4)
# fit <- model$estimate(X, threads = 4)

summary(fit, extended = TRUE)
print(round(fit$xm, 3))


## Calculate the one-step predictions of the state (i.e. the residuals)
tmp <- predict(fit, n.ahead = 1)[[1]]
## Calculate the residuals and put them with the data in a data.frame X
X$residualsTwu <- X$yTwu - tmp$output$pred$yTwu
X$residualsTwl <- X$yTwl - tmp$output$pred$yTwl
X$TwuHat <- tmp$output$pred$yTwu
X$TwlHat <- tmp$output$pred$yTwl

## Plot the auto-correlation function and cumulated periodogram in a new window
filename <- paste(BASE_FOLDER, "4thOrderModelValidation.png", sep = "")
png(filename = filename, width = 1200, height = 800, res = 300)
par(mfrow = c(2, 3), mai = c(0.5, 0.5, 0.5, 0.5), mar = c(4, 4, 2, 1))
## The blue lines indicates the 95 confidence interval, meaning that if it is
## white noise, then approximately 1 out of 20 lag correlations will be slightly outside
acf(X$residualsTwu, lag.max = 6 * 12, main = "Residuals ACF")
spec.pgram(X$residualsTwu, main = "Raw periodogram")
cpgram(X$residualsTwu, main = "Cumulated periodogram")
acf(X$residualsTwl, lag.max = 6 * 12, main = "Residuals ACF")
spec.pgram(X$residualsTwl, main = "Raw periodogram")
cpgram(X$residualsTwl, main = "Cumulated periodogram")
dev.off()

# Simulate ODE
Czu <- fit$xm["Czu"]
Czl <- fit$xm["Czl"]
Cwu <- fit$xm["Cwu"]
Cwl <- fit$xm["Cwl"]
Rww <- fit$xm["Rww"]
Rwz <- fit$xm["Rwz"]
Rza <- fit$xm["Rza"]
Rzuzl <- fit$xm["Rzuzl"]
Rwua1 <- fit$xm["Rwua1"]
Rwua2 <- fit$xm["Rwua2"]
Rwla1 <- fit$xm["Rwla1"]
# Rwla2 <- fit$xm["Rwla2"]

##
lb <- 6400
ub <- 7800
# lb = 1000
# ub = 2000
X2 <- XCopy[lb:ub, ]

Tzu_sim <- c(dim(X2)[1])
Tzl_sim <- c(dim(X2)[1])
Twu_sim <- c(dim(X2)[1])
Twl_sim <- c(dim(X2)[1])
Tzu_sim[1] <- setpoint_u
Tzl_sim[1] <- setpoint_l
Twu_sim[1] <- X2$yTwu[1]
Twl_sim[1] <- X2$yTwl[1]

Ptu_ <- c(dim(X2)[1])
Ptl_ <- c(dim(X2)[1])
Pt_ <- c(dim(X2)[1])
Ptl_[1] <- X2$PtLow[1]
Ptu_[1] <- X2$PtUp[1]
Pt_[1] <- Ptl_[1] + Ptu_[1]

if (Ptl_[1] > 400) {
  Q1l <- 1
  Q2l <- 1
}
if (Ptl_[1] < 400 & Ptl_[1] > 90) {
  Q1l <- 1
  Q2l <- 0
}
if (Ptl_[1] < 90) {
  Q1l <- 0
  Q2l <- 0
}
if (Ptu_[1] > 400) {
  Q1u <- 1
  Q2u <- 1
}
if (Ptu_[1] < 400 & Ptu_[1] > 90) {
  Q1u <- 1
  Q2u <- 0
}
if (Ptu_[1] < 90) {
  Q1u <- 0
  Q2u <- 0
}

for (i in 2:dim(X2)[1]) {
  # simulate contactor logic
  if (Twl_sim[i - 1] < 446.5) {
    Q1l <- 1
  } else if (Twl_sim[i - 1] > 448.5) {
    Q1l <- 0
  }
  if (Twl_sim[i - 1] < 445.5) {
    Q2l <- 1
  } else if (Twl_sim[i - 1] > 447.5) {
    Q2l <- 0
  }
  if (Q1l + Q2l == 0) {
    Ptl <- 0
  } else if (Q1l + Q2l == 1) {
    Ptl <- 111
  } else if (Q1l + Q2l == 2) {
    Ptl <- 442
  }

  if (Twu_sim[i - 1] < 454.5) {
    Q1u <- 1
  } else if (Twu_sim[i - 1] > 456.5) {
    Q1u <- 0
  }
  if (Twu_sim[i - 1] < 453.5) {
    Q2u <- 1
  } else if (Twu_sim[i - 1] > 455.5) {
    Q2u <- 0
  }
  if (Q1u + Q2u == 0) {
    Ptu <- 0
  } else if (Q1u + Q2u == 1) {
    Ptu <- 111
  } else if (Q1u + Q2u == 2) {
    Ptu <- 442
  }

  # see what happens when turning power off
  if (i >= 430 & i < 500) {
    Ptu <- 0
    Ptl <- 0
  }

  Ptl_[i] <- Ptl
  Ptu_[i] <- Ptu
  Pt_[i] <- Ptl_[i] + Ptu_[i]

  # Tzu_sim[i] <- Tzu_sim[i - 1] + 1 / Czu * dt_ * (1 / Rzuzl * (Tzl_sim[i - 1] - Tzu_sim[i - 1]) + 1 / Rwz * (Twu_sim[i - 1] - Tzu_sim[i - 1]) + X2$reg[i] / Rza * (Ta - Tzu_sim[i - 1]))
  Tzu_sim[i] <- Tzu_sim[i - 1] + 1 / Czu * dt_ * (1 / Rzuzl * (Tzl_sim[i - 1] - Tzu_sim[i - 1]) + 1 / Rwz * (Twu_sim[i - 1] - Tzu_sim[i - 1]))
  Tzl_sim[i] <- Tzl_sim[i - 1] + 1 / Czl * dt_ * (1 / Rzuzl * (Tzu_sim[i - 1] - Tzl_sim[i - 1]) + 1 / Rwz * (Twl_sim[i - 1] - Tzl_sim[i - 1]))

  Twu_sim[i] <- Twu_sim[i - 1] + 1 / Cwu * dt_ * ((1 - X2$reg[i]) * 1 / Rwua1 * (20 - Twu_sim[i - 1]) + 1 / Rww * (Twl_sim[i - 1] - Twu_sim[i - 1]) + 1 / Rwz * (Tzu_sim[i - 1] - Twu_sim[i - 1]) + X2$reg[i] / Rwua2 * (Ta - Twu_sim[i - 1]) + Ptu)
  Twl_sim[i] <- Twl_sim[i - 1] + 1 / Cwl * dt_ * (1 / Rwla1 * (20 - Twl_sim[i - 1]) + 1 / Rww * (Twu_sim[i - 1] - Twl_sim[i - 1]) + 1 / Rwz * (Tzl_sim[i - 1] - Twl_sim[i - 1]) + Ptl)
}

X2$Pt <- X2$PtUp + X2$PtLow

print(sum(Ptu_) / 60)
print(sum(X2$PtUp) / 60)
print(abs((sum(X2$PtUp) - sum(Ptu_)) / 60) / (sum(X2$PtUp) / 60) * 100)

print(sum(Ptl_) / 60)
print(sum(X2$PtLow) / 60)
print(abs((sum(X2$PtLow) - sum(Ptl_)) / 60) / (sum(X2$PtLow) / 60) * 100)

par(mfrow = c(3, 1), mai = c(0.5, 0.9, 0.5, 0.9))
plot(X2$t[1:length(X2$t)], X2$yTwu[1:length(X2$t)], type = "n", ylab = "Celsius", ylim = c(440, 460))
lines(X2$t[1:length(X2$t)], X2$yTwu[1:length(X2$t)], col = 1)
lines(X2$t[1:length(X2$t)], Twu_sim, col = 2, lty = 2)
lines(X2$t[1:length(X2$t)], Tzu_sim, col = 3, lty = 2)
abline(h = setpoint_u, col = "black", lty = 2)
legend("bottomright", c("Measured", "Twu_sim", "Tzu_sim"), col = 1:3, lty = 1)


plot(X2$t[1:length(X2$t)], X2$yTwl[1:length(X2$t)], type = "n", ylab = "Celsius", ylim = c(440, 452))
lines(X2$t[1:length(X2$t)], X2$yTwl[1:length(X2$t)], col = 1)
lines(X2$t[1:length(X2$t)], Twl_sim, col = 2, lty = 2)
lines(X2$t[1:length(X2$t)], Tzl_sim, col = 3, lty = 2)
# lines(X2$t[1:length(X2$t)], Tw_sim, col=3)
abline(h = setpoint_l, col = "black", lty = 2)
legend("bottomright", c("Measured", "Twl_sim", "Tzl_sim"), col = 1:3, lty = 1)

plot(X2$t, Pt_, type = "n", ylab = "KW")
lines(X2$t, Pt_, col = 1)
lines(X2$t, Ptu_, col = 2, lty = 2)
lines(X2$t, Ptl_, col = 3, lty = 2)
legend("topleft", c("Total", "KW_UZ", "KW_LZ"), col = 1:3, lty = 1)
