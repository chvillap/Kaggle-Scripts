# =============================================================================
# analysis.R
#
# Script for doing some exploratory data analysis.
# =============================================================================

# -----------------------------------------------------------------------------
# Basic information (variable types, NAs, statistics, ...).

library(data.table)

df1 <- as.data.frame(fread("datasets/training.csv"))
str(df1)
summary(df1)


# -----------------------------------------------------------------------------
# Distribution of keypoint coordinate values.

library(dplyr)
library(tidyr)
library(ggplot2)

df2 <- gather(df1[, -31], keypoint, value, -31)
df2 <- df2 %>%
       arrange(keypoint) %>%
       subset(!is.na(value))

palette <- c("red", "green", "blue", "yellow", "purple", "turquoise", "black",
             "orange", "deeppink", "greenyellow", "cadetblue", "darkgray",
             "saddlebrown", "tomato", "darkolivegreen4")

ggplot(df2, aes(x = keypoint, y = value, colour = keypoint)) +
    geom_boxplot(outlier.shape = 4) +
    ggtitle("Keypoint positions: coordinate values") +
    guides(colour = FALSE) +
    scale_colour_manual(values = as.vector(sapply(palette, rep, 2))) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1),
          plot.title = element_text(size = 14, face = "bold"))

# -----------------------------------------------------------------------------
# Scatter plot of keypoint positions.

df3 <- mutate(df2, axis = substr(keypoint, nchar(keypoint), nchar(keypoint)))
df3$keypoint <- substr(df3$keypoint, 1, nchar(df3$keypoint)-2)
df3x <- df3 %>%
        filter(axis == "x") %>%
        mutate(x = value) %>%
        select(c(keypoint, x))
df3y <- df3 %>%
        filter(axis == "y") %>%
        mutate(y = value) %>%
        select(y)
df3 <- cbind.data.frame(df3x, df3y)

ggplot(df3, aes(x = x, y = -y, colour = keypoint)) +
    geom_point(shape = 4) +
    ggtitle("Keypoint positions: all") +
    scale_colour_manual(values = palette) +
    coord_cartesian(xlim = c(1, 96), ylim = c(-96, -1)) +
    theme(plot.title = element_text(size = 14, face = "bold"))

# -----------------------------------------------------------------------------
# More scatter plots, now one for each keypoint separately.

knames <- unique(df3$keypoint)
for (i in 1:length(knames)) {
    df4 <- filter(df3, keypoint == knames[i])
    g <- ggplot(df4, aes(x = x, y = -y, colour = keypoint)) +
             geom_point(shape = 4) +
             ggtitle(paste("Keypoint positions: ", knames[i])) +
             scale_colour_manual(values = palette[i]) +
             coord_cartesian(xlim = c(1, 96), ylim = c(-96, -1)) +
             theme(plot.title = element_text(size = 14, face = "bold"))
    print(g)
}

# -----------------------------------------------------------------------------
# Correlation map of keypoint positions.

library(lattice)

cor_mat <- cor(df1[, -31], method = "pearson", use = "complete")
cor_mat

levelplot(cor_mat,
          main = "Keypoint positions: correlation map",
          scales = list(x = list(rot = 90), cex = 1))
