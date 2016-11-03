# =============================================================================
# eda_clean.R
#
# Script to do some exploratory data analysis and also clean the data sets.
# =============================================================================

library(dplyr)
library(randomForest)
library(ggplot2)
library(lattice)
library(RColorBrewer)

# -----------------------------------------------------------------------------

df <- read.csv("datasets/train.csv")

# Basic info (sizes, names, types, missing values, some stats).
str(df)
summary(df)
head(df)
tail(df)

# Count missing, blank and unique values.
df.cnt <- data.frame(
    na.cnt     = sapply(df, function(x) {sum(is.na(x))}),
    blank.cnt  = sapply(df, function(x) {sum(!is.na(x) & x == "")}),
    unique.cnt = sapply(df, function(x) {length(unique(x))}))
df.cnt <- df.cnt %>%
          mutate(na.perc     = round(100 * na.cnt / nrow(df), 2),
                 blank.perc  = round(100 * blank.cnt / nrow(df), 2),
                 unique.perc = round(100 * unique.cnt / nrow(df), 2))
rownames(df.cnt) <- colnames(df)
df.cnt <- df.cnt[, c(1, 4, 2, 5, 3, 6)]
df.cnt

rm(df)
rm(df.cnt)

# Some problems already detected:
#   PassengerId ==> ID variable (should be removed).
#   Survived    ==> Not a factor (categorical) variable.
#   Name        ==> Title, and maybe Age, can be inferred from it.
#   Age         ==> 177 (~19.87%) missing values.
#   Ticket      ==> 681 (~76.43%) unique values.
#   Cabin       ==> 687 (~77.10%) blank values.
#   Embarked    ==> 2 (~0.45%) blank values.
#
# Additional notes:
#   Name ==> contents contain quotation marks and parentheses, but no problem.
#   Other variables are ok.

# -----------------------------------------------------------------------------

# Some functions to plot charts using ggplot.

plot.bar <- function(data, x, fill) {
    ggplot(data, aes_string(x = x, fill = fill)) +
        scale_fill_brewer(palette = "Paired") +
        geom_bar() +
        geom_text(aes(label = ..count..),
                  stat = "count",
                  vjust = -0.5,
                  position = "identity")
}

plot.stacked.bar <- function(data, x, fill, val.pos, val.neg) {
    ggplot(data, aes_string(x = x, fill = fill)) +
        scale_fill_brewer(palette = "Paired") +
        geom_bar(data = subset(data, Survived == val.pos)) +
        geom_bar(data = subset(data, Survived == val.neg),
                 aes(y = -1 * ..count..)) +
        geom_text(data = subset(data, Survived == val.pos),
                  aes(label = ..count..),
                  stat = "count",
                  vjust = -0.5) +
        geom_text(data = subset(data, Survived == val.neg),
                  aes(y = -1 * ..count.., label = ..count..),
                  stat = "count",
                  vjust = 1.5) +
        guides(fill = guide_legend(reverse = TRUE))
}

plot.histogram <- function(data, x, fill, bins = 10) {
    ggplot(data, aes_string(x = x, fill = fill)) +
        scale_fill_brewer(palette = "Paired") +
        geom_histogram(bins = bins) +
        facet_grid(reformulate(".", fill))
}

plot.box.violin <- function(data, x, y, fill, violin = TRUE) {
    if (violin) {
        ggplot(data, aes_string(x = x, y = y, fill = fill)) +
            scale_fill_brewer(palette = "Paired") +
            geom_violin() +
            geom_boxplot(width = 0.1, outlier.stroke = NA) +
            stat_summary(fun.y = "mean", geom = "point", shape = 1)
    } else {
        ggplot(data, aes_string(x = x, y = y, fill = fill)) +
            scale_fill_brewer(palette = "Paired") +
            geom_boxplot(outlier.shape = 4) +
            stat_summary(fun.y = "mean", geom = "point", shape = 1)
    }
}

# -----------------------------------------------------------------------------

fnames <- c("train", "test")

for (f in fnames) {
    # Blanks are now treated as missing values too.
    df <- read.csv(sprintf("datasets/%s.csv", f), na.strings = c(""))

    # Create a new Title variable.
    # The title is extracted from the passenger's name.
    title.pattern <- ",\\s([[:alnum:]]+)."
    df <- mutate(df, Title = regmatches(Name, gregexpr(title.pattern, Name)))
    df$Title <- substr(df$Title, 3, nchar(df$Title) - 1)

    # By counting how many times each Title value appear, we can see that some
    # are so rare that we should unify them into a single value.
    print(table(df$Title))
    df$Title <- sapply(df$Title, function (x) {
        ifelse(x %in% c("Dr", "Master", "Miss", "Mr", "Mrs", "Rev"), x, "rare")
    })
    print(table(df$Title))

    # Create a new CabinDeck variable from Cabin.
    # The cabin deck is the first letter of the cabin name.
    df <- mutate(df, CabinDeck = substr(Cabin, 1, 1))
    print(table(df$CabinDeck))

    print(str(df))
    print(summary(df))

    # -------------------------------------------------------------------------

    # Treat the target variable as categorical.
    if (f == "train") {
        df$Survived <- as.factor(df$Survived)

        # Plot some charts to take a look at our data (with missing values).
        print(plot.bar(df, "Survived", "Survived"))
        print(plot.box.violin(df, "Survived", "Age", "Survived"))
        print(plot.box.violin(df, "Survived", "Fare", "Survived"))
        print(plot.histogram(df, "SibSp", "Survived"))
        print(plot.histogram(df, "Parch", "Survived"))
        print(plot.stacked.bar(df, "Pclass", "Survived", 1, 0))
        print(plot.stacked.bar(df, "Sex", "Survived", 1, 0))
        print(plot.stacked.bar(df, "Embarked", "Survived", 1, 0))
        print(plot.stacked.bar(df, "Title", "Survived", 1, 0))
        print(plot.box.violin(df, "Title", "Age", "Sex", violin = FALSE))
        print(plot.stacked.bar(df, "CabinDeck", "Survived", 1, 0))
    }

    # -------------------------------------------------------------------------

    # Title has a high correlation with Sex and, in some cases, with Age.
    # So the missing Age values are filled with the median relative to Title.
    for (title in unique(df$Title)) {
        df[is.na(df$Age) & df$Title == title, "Age"] <-
            median(df[!is.na(df$Age) & df$Title == title, "Age"])
    }
    
    # The missing Embarked values are filled with the mode.
    md <- which.max(table(df$Embarked))
    df[is.na(df$Embarked), "Embarked"] <- names(md)

    # The missing Fare values are filled with the median relative to Pclass.
    pclass <- df[is.na(df$Fare), "Pclass"]
    df[is.na(df$Fare), "Fare"] <-
        median(df[!is.na(df$Fare) & df$Pclass == pclass, "Fare"])

    # Most (~77.10%) CabinDeck values are actually missing, so it's hard to
    # try to estimate them. Better to fill them with a distinct arbitrary value
    # (U). Moreover, CabinDeck has, in principle, 8 levels in the training set,
    # but only 7 in the test set. As there's only 1 observation with the
    # additional value (T), it is going to be replaced too.
    df$CabinDeck[is.na(df$CabinDeck)] <- "U"
    df[df$CabinDeck == "T", "CabinDeck"] <- "U"

    print(str(df))
    print(summary(df))

    # -------------------------------------------------------------------------

    # Create a new StageOfLife variable from Age.
    # 1 = child, 2 = teen, 3 = adult, 4 = elder.
    # It could be represented as an ordered factor as well.
    df$StageOfLife <- ifelse(df$Age < 12, 1,
                             ifelse(df$Age < 21, 2,
                                    ifelse(df$Age < 50, 3, 4)))

    # Do some feature engineering to try to increase the discriminative power
    # of certain combinations of variables.
    df <- mutate(df, SexPol = ifelse(df$Sex == "female", -1, 1))
    df <- mutate(df, Relatives = SibSp + Parch)
    df <- mutate(df, "SexPol_mul_Age" = SexPol * Age)
    df <- mutate(df, "SexPol_mul_Pclass" = SexPol * Pclass)
    df <- mutate(df, "Age_mul_Pclass2" = Age * Pclass^2)
    df <- mutate(df, "root_Fare_div_Pclass" = sqrt(Fare) / Pclass)

    print(str(df))
    print(summary(df))

    # Plot charts for the new features. It is visible that some of them seem
    # to improve the discriminative power of survivors.
    if (f == "train") {
        print(plot.stacked.bar(df, "StageOfLife", "Survived", 1, 0))
        print(plot.box.violin(df, "Survived", "SexPol_mul_Age", "Survived"))
        print(plot.box.violin(df, "Survived", "Age_mul_Pclass2", "Survived"))
        print(plot.box.violin(df, "Survived", "root_Fare_div_Pclass", "Survived"))
        print(plot.histogram(df, "Relatives", "Survived"))
        print(plot.stacked.bar(df, "SexPol_mul_Pclass", "Survived", 1, 0))
    }

    # -------------------------------------------------------------------------
    
    as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}

    if (f == "train") {
        # Drop categorical variables.
        df.num <- subset(df, select = -c(PassengerId, Name, Sex, Ticket, Cabin,
                                         Embarked, Title, CabinDeck))
        df.num$Survived <- as.numeric.factor(df.num$Survived)
    
        # Plot the correlation map between all numerical variables.
        brewer.div <- colorRampPalette(brewer.pal(11, "RdBu"))#, interpolate = "spline")
        print(levelplot(cor(df.num),
                        cuts = 100,
                        col.regions = brewer.div,
                        scales = list(x = list(rot = 90))))
    }

    # -------------------------------------------------------------------------

    # Set remaining categorical variables to factors.
    df$Title <- as.factor(df$Title)
    df$CabinDeck <- as.factor(df$CabinDeck)

    # Drop unnecessary variables.
    # Three different versions are tried: one with both original and combined
    # features, one with the original features only, and a third one with the
    # combined features only.
    df.sub <- list(
        subset(df, select = -c(Name, Ticket, Cabin, Sex)),
        subset(df, select = -c(Name, Ticket, Cabin, Sex,
                               Pclass, Sex, Age, SibSp, Parch, Fare)),
        subset(df, select = -c(Name, Ticket, Cabin, Sex,
                               Relatives, SexPol_mul_Age, SexPol_mul_Pclass,
                               Age_mul_Pclass2, root_Fare_div_Pclass)),
        subset(df, select = -c(Name, Ticket, Cabin, Sex, Age, SibSp, Parch,
                               SexPol_mul_Age, SexPol_mul_Pclass,
                               Age_mul_Pclass2, root_Fare_div_Pclass)))

    if (f == "train") {
        lapply(df.sub, function(x) {
            print(str(x))
            print(summary(x))})

        # Use Random Forest models to quantify the importance of each variable.
        # The PassengerId variable is ignored here.
        rf <- lapply(df.sub, function(x) {
                randomForest(Survived ~ .,
                             data = subset(x, select = -PassengerId),
                             ntree = 200, nodesize = 5, importance = TRUE)})

        # Check the results.
        lapply(rf, function(x) {
            print(x)
            plot(x)
            varImpPlot(x)})
    }

    # -------------------------------------------------------------------------

    # Write the cleaned data sets.
    for (k in 1:length(df.sub)) {
        fname <- sprintf("%s%d.csv", f, k)
        write.csv(df.sub[[k]], fname, quote = FALSE, row.names = FALSE)
    }
}
