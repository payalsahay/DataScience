#-------------------------------Begin-------------------------------#

# load libraries
library(ggplot2)
library(dplyr)
library(VIM)
library(data.table)
library(caret)
library(dummies)
library(cluster)
library(dendextend)
library(datasets)
library(NbClust)
library(heatmaply)
library(kohonen)
library(fpc)
library(dbscan)
library(factoextra)
library(tidyr)
library(RColorBrewer)
library(gridExtra)

#Read telecom data from telco-customer-churn.csv
data <- read.csv("~/Desktop/620 - Data Mining/Assignment 4/WA_Fn-UseC_-Telco-Customer-Churn.csv", na = " ") 
View(data)
str(data)

#-------------------------------Part 1-------------------------------#

#copy in a new dataframe
Data <- data 

#check for missing values
summary(aggr(Data, plot =  TRUE))

#since there are very few rows with missing values in the total charges we drop them
Data <- na.omit(Data)
summary(aggr(Data, plot =  TRUE))

#check for duplicate rows
summary(duplicated(Data$customerID))

#assign customerID as rownames
rownames(Data) <- Data$customerID
head(Data)

#get factor variables
names(Data[,sapply(Data, is.factor)])

#drop column customerID
Data$customerID <- NULL
str(Data)

#convert everything to numeric but first copy in a new dataframe
NumData <- Data
NumData[,1:20] <- lapply(NumData[,1:20], as.numeric)
View(NumData)

#scale data 
ScaledData <- scale(NumData)
head(ScaledData)
summary(ScaledData)

#-------distance matrices-----------#
#calculate distance matrix on scaled data

EucDist <- dist(ScaledData, method = "euclidean")
summary(EucDist)

ManDist <- dist(ScaledData, method = "manhattan")
summary(ManDist)

MinDist <- dist(ScaledData, method = "minkowski")
summary(MinDist)

MaxDist <- dist(ScaledData, method = "maximum")
summary(MaxDist)

#---------Hierarchical Clustering---------#
# Euclidean distance
HClustEuc <- hclust(EucDist)
plot(HClustEuc)

#cut tree at h = 9.5 to get a reduced number of clusters
EucDend <- as.dendrogram(HClustEuc, cex= 0.5, hang = 0.1)
plot(cut(EucDend, h = 9.5)$upper, main = "Upper cut of tree at h=9.5")

# Manhattan distance
HClustMan <- hclust(ManDist)
plot(HClustMan)

#cut tree at h = 39 to get a reduced number of clusters
ManDend <- as.dendrogram(HClustMan, cex= 0.5, hang = 0.1)
plot(cut(ManDend, h = 39)$upper, main = "Upper cut of tree at h=39")

# Min distance
HClustMin <- hclust(MinDist)
plot(HClustMin)

#cut tree at h = 9.5 to get a reduced number of clusters
MinDend <- as.dendrogram(HClustMin, cex= 0.5, hang = 0.1)
plot(cut(MinDend, h = 9.5)$upper, main = "Upper cut of tree at h=9.5")

# Max distance
HClustMax <- hclust(MaxDist)
plot(HClustMax)

#cut tree at h = 3 to get a reduced number of clusters
MaxDend <- as.dendrogram(HClustMax, cex= 0.5, hang = 0.1)
plot(cut(MaxDend, h = 3)$upper, main = "Upper cut of tree at h=3")

#------------AGNES Clustering------------#

AgnesClust <- agnes(ScaledData, metric = "euclidean")

# See the clusters
AgnesClust$ac
summary(AgnesClust$order)
summary(AgnesClust$height)

#get Dendrogram
AgnesDend <-as.dendrogram(AgnesClust, cex= 0.5, hang = 0.1)

#plot dendrogram
plot(AgnesDend)
plot(cut(AgnesDend, h = 6)$upper, main = "Upper cut of tree at h=6")

#get height for each observation
heights_per_k.dendrogram(AgnesDend)

#------------Kohonen Self Organizing Map-------------#
# Create Kohonen SOM in two parts of 10 attributes each
KhnnClust <- som(ScaledData[,1:10], grid = somgrid(5, 4, "hexagonal"))
SomClust <- som(ScaledData[,11:20], grid = somgrid(5, 4, "hexagonal"))

# Plot map
plot(KhnnClust, type="codes", labels = rownames(ScaledData))
plot(KhnnClust, type="mapping", labels = rownames(ScaledData))

plot(SomClust, type="codes", labels = rownames(ScaledData))
plot(SomClust, type="mapping", labels = rownames(ScaledData))

#-------------Density Based Clustering--------------#
DBClust <- dbscan(ScaledData, eps = 3, minPts = 30, border = TRUE)
print(DBClust)

# see clusters
DBClust$eps
DBClust$minPts
DBClust$cluster

# plot clusters
plot(DBClust, ScaledData, main = "DBSCAN")

#-----------------k-means Clustering---------------------#
#comparison of k
k2 <- kmeans(ScaledData, centers = 2, nstart = 25)
k3 <- kmeans(ScaledData, centers = 3, nstart = 25)
k4 <- kmeans(ScaledData, centers = 4, nstart = 25)
k5 <- kmeans(ScaledData, centers = 5, nstart = 25)
k6 <- kmeans(ScaledData, centers = 6, nstart = 25)

# create plots for comparison
p2 <- fviz_cluster(k2, geom = "point", data = ScaledData) + ggtitle("k = 2")
p3 <- fviz_cluster(k3, geom = "point", data = ScaledData) + ggtitle("k = 3")
p4 <- fviz_cluster(k4, geom = "point", data = ScaledData) + ggtitle("k = 4")
p5 <- fviz_cluster(k5, geom = "point", data = ScaledData) + ggtitle("k = 5")
p6 <- fviz_cluster(k6, geom = "point", data = ScaledData) + ggtitle("k = 6")

# plot in a grid
grid.arrange(p2, p3, p4, p5, p6, nrow = 3)

# check by NbClust the best number of k
KNum <- NbClust(ScaledData, method="kmeans")
KNum$Best.nc

# create elbow plot for the optimal number of clusters
fviz_nbclust(ScaledData, kmeans, method = "wss") 

# since 3 is the optimal k (firs dip is at 3), perform k-means
KmeansClust <- kmeans(ScaledData, 3, nstart = 25)
plot(Data[,c(1:5)], col = KmeansClust$cluster)
KmeansClust
fviz_cluster(KmeansClust, data = ScaledData)

#See the clusters
KmeansClust$cluster
table(KmeansClust$cluster)
KmeansClust$size

# Get sum of squares within clusters and between clusters
KmeansClust$withinss
KmeansClust$tot.withinss
KmeansClust$betweenss

#combine original data with respective cluster numbers
NewData <- cbind(Data, KmeansClust$cluster)
class(NewData)
View(NewData)

#rename the new column as cluster
colnames(NewData)[21] <- "Cluster"

#--------------------------------------Part 2--------------------------------------------#

#--------------2a.	Segment size (percentage)------------------#
SegSize <- KmeansClust$size
SegPercent <- SegSize[1:3]/sum(SegSize)*100
Segment <- c("Segment1", "Segment2", "Segment3")
print(data.frame(Segment, SegSize, SegPercent))

#-------2b.	Segment characteristics (important features)-------#

#create a cluster index
KmCluster <- c(1:3)

#read cluster centers to interpret cluster characteristics
ClustCenter <- KmeansClust$centers

#divide cluster centers in two parts each for 10 attributes
CenterDF1 <- data.frame(KmCluster, ClustCenter[,1:10])
CenterDF2 <- data.frame(KmCluster, ClustCenter[,11:20])

#prepare cluster centers for heatmap plotting
CenterReshape1 <- gather(CenterDF1, features, values, gender: OnlineBackup)
head(CenterReshape1)

CenterReshape2 <- gather(CenterDF2, features, values, DeviceProtection: Churn)
head(CenterReshape2)

#create the palette
HeatMapPalette <- colorRampPalette(rev(brewer.pal(10, 'RdYlGn')),space='Lab')

#create the heat maps in two parts
hm1 <- ggplot(data = CenterReshape1, aes(x = features, y = KmCluster, fill = values)) +
  scale_y_continuous(breaks = seq(1, 7, by = 1)) +
  geom_tile() +
  coord_equal() +
  scale_fill_gradientn(colours = HeatMapPalette(90)) +
  theme_classic()

hm2 <- ggplot(data = CenterReshape2, aes(x = features, y = KmCluster, fill = values)) +
  scale_y_continuous(breaks = seq(1, 7, by = 1)) +
  geom_tile() +
  coord_equal() +
  scale_fill_gradientn(colours = HeatMapPalette(90)) +
  theme_classic()

# plot heat maps for both parts
grid.arrange(hm1, hm2, nrow = 2)

#------------2c.	Segment revenue share (% of revenue)---------------#
#divide data into segments and calculate revenue using TotalCharges
Seg1 <- subset(NewData, NewData$Cluster == "1")
head(Seg1)
nrow(Seg1)
Rev1 <- sum(Seg1$TotalCharges)
Rev1

Seg2 <- subset(NewData, NewData$Cluster == "2")
head(Seg2)
nrow(Seg2)
Rev2 <- sum(Seg2$TotalCharges)
Rev2

Seg3 <- subset(NewData, NewData$Cluster == "3")
head(Seg3)
nrow(Seg3)
Rev3 <- sum(Seg3$TotalCharges)
Rev3

# calculate total revenue
TotalRev <- sum(Data$TotalCharges)
TotalRev
print(sum(Rev1, Rev2, Rev3))

#calulate revenue percentage
Seg1Share <- Rev1/TotalRev*100
Seg2Share <- Rev2/TotalRev*100
Seg3Share <- Rev3/TotalRev*100

#print segment revenue percentage
SegRevShare <- c(Seg1Share, Seg2Share, Seg3Share)
print(data.frame(Segment, SegRevShare))

#---------2d.	Percent of segment customers at risk of leaving----------#
Seg1Leave <- subset(Seg1, Seg1$Churn == "Yes")
Seg1LeavePer <- nrow(Seg1Leave)/nrow(Seg1)*100

Seg2Leave <- subset(Seg2, Seg2$Churn == "Yes")
Seg2LeavePer <- nrow(Seg2Leave)/nrow(Seg2)*100

Seg3Leave <- subset(Seg3, Seg3$Churn == "Yes")
Seg3LeavePer <- nrow(Seg3Leave)/nrow(Seg3)*100

#print percentage for customers at risk of leaving
SegLeaveRisk <- c(Seg1LeavePer, Seg2LeavePer, Seg3LeavePer)
print(data.frame(Segment, SegLeaveRisk))

#----------------2e.	Revenue at risk per segment-------------------#
RevRisk1 <- sum(Seg1Leave$TotalCharges)
RevRisk2 <- sum(Seg2Leave$TotalCharges)
RevRisk3 <- sum(Seg3Leave$TotalCharges)

#print revenue at risk per segment
SegRevRisk <- c(RevRisk1, RevRisk2, RevRisk3)
print(data.frame(Segment, SegRevRisk))

#total revenue at risk
TotalRevRisk <- sum(RevRisk1, RevRisk2, RevRisk3)
TotalRevRisk

#percentage of total revenue at risk per segment
RiskPer1 <- RevRisk1/TotalRevRisk*100
RiskPer2 <- RevRisk2/TotalRevRisk*100
RiskPer3 <- RevRisk3/TotalRevRisk*100

#print percentage of total revenue at risk per segment
SegRiskShare <- c(RiskPer1, RiskPer2, RiskPer3)
print(data.frame(Segment, SegRiskShare))

#percentage of total revenue at risk
RevRiskPer <- TotalRevRisk/TotalRev*100
RevRiskPer

#-------------------------------End-------------------------------#