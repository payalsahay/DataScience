#--------------begin--------------#

#load library for reading excel file
library(readxl)
#load library arules
library(arules)

#read excel file in a dataframe
T_List <- read_excel("~/Desktop/620 - Data Mining/Assignment 2/TransactionList.xlsx")
head(T_List) #view top rows
class(T_List) #view the type of imported data
summary(T_List) #view summary of the dataframe
View(T_List) #view the data in a separate window

#write in a csv file
write.csv(T_List, "~/Desktop/620 - Data Mining/Assignment 2/T_List.csv", row.names = FALSE)

#read from the csv file in transactions format
T_Data <- read.transactions("~/Desktop/620 - Data Mining/Assignment 2/T_List.csv",
                            format = "single", sep = ",", 
                            cols = c("transaction_id", "product name"), 
                            rm.duplicates = TRUE)
head(T_Data) #view top rows
class(T_Data) #view the type of imported data
View(T_Data) #view the data in a separate window
summary(T_Data) #view summary of the transaction data

#plot frequencies of frequent items
itemFrequencyPlot(T_Data, support=0.1, cex.names=1.0)

# Run ariori algorithm and generate rules
Rules_T <- apriori(T_Data, parameter = list(support=0.01, confidence=0.5, 
                                            minlen=2, target='rules'))
summary(Rules_T)
quality(Rules_T)$lift <- interestMeasure(Rules_T, measure='lift', T_Data)

#calculate interest measures and add it to the quality slot
quality(Rules_T) <- cbind(quality(Rules_T), 
                          kulc = interestMeasure(Rules_T, measure = "kulczynski", 
                                                 transactions = T_Data))
#calculate chi interest measure 
quality(Rules_T)$chi <- interestMeasure(Rules_T, measure='chi', significance=T, T_Data)

# calculate confidence 
inspect(sort(Rules_T, by="confidence", decreasing = T)[1:10])

#display interest measures, support, confidence, lift and count
quality(head(Rules_T))

#check for positive correaltions
inspect(sort(Rules_T, decreasing = TRUE, by='lift')[1:10])
inspect(head(Rules_T, by = "kulc"))

#check for negative correaltions
inspect(sort(Rules_T, decreasing = FALSE, by='lift')[1:10]) 
inspect(tail(Rules_T, by = "kulc")) 

#calculate all available measures for the first 10 rules and show them as a 
#table with the measures as rows
t(interestMeasure(head(Rules_T, 10), transactions = T_Data))

#---------Visualization of Transaction Rules---------#

#load library for visualization
library(arulesViz)

#plot between support and confidence with lift shading
plot(Rules_T)

#plot between support and lift with confidence shading
plot(Rules_T, measure=c("support", "lift"), shading="confidence")

# Graph visualization
T_Rules <- head(sort(Rules_T, by="lift"), 10)

plot(T_Rules, method="graph")
plot(T_Rules, method="graph", control=list(type="itemsets"))

# parallel coordinates, width is support, color is confidence
plot(T_Rules, method="paracoord")

#--------------Part 1----------------#

#we find subset of rules that has Wine in the Right hand side
WineRulesRHS <- subset(Rules_T, subset = rhs %in% "Wine")
inspect(sort(WineRulesRHS, by = "lift"))
summary(WineRulesRHS)
plot(WineRulesRHS, method = "graph")

#we find subset of rules that has Wine in the Left hand side
WineRulesLHS <- subset(Rules_T, subset = lhs %pin% "Wine")
inspect(sort(WineRulesLHS, by = "lift"))
summary(WineRulesLHS)
plot(WineRulesLHS, method = "graph")

#we find subset of rules that has Beer in the Right hand side
BeerRulesRHS <- subset(Rules_T, subset = rhs %pin% "Beer")
summary(BeerRulesRHS)

#we find subset of rules that has Beer in the Left hand side
BeerRulesLHS <- subset(Rules_T, subset = lhs %pin% "Beer")
summary(BeerRulesLHS)
#no rules related to Beer that satisfy the minimum support and confidence threshold

#we find subset of rules that has Soda in the Right hand side
SodaRulesRHS <- subset(Rules_T, subset = rhs %in% "Soda")
inspect(sort(SodaRulesRHS, by = "lift"))
summary(SodaRulesRHS)
plot(SodaRulesRHS, method = "graph")

#we find subset of rules that has Soda in the Left hand side 
SodaRulesLHS <- subset(Rules_T, subset = lhs %in% "Soda")
inspect(sort(SodaRulesLHS, by = "lift")[1:5])
summary(SodaRulesLHS)
plot(SodaRulesLHS)

#we find subset of rules that has Wine or Soda in RHS
WineSodaRules <- subset(Rules_T, subset = rhs %in% c("Wine", "Soda"))
inspect(WineSodaRules)
summary(WineSodaRules)
plot(WineSodaRules, method = "graph")
#there are no common things that people buy that initiate the buying of wine and soda

#we find subset of rules that has Wine or Soda in LHS
WSRules <- subset(Rules_T, subset = lhs %ain% c("Soda","Wine"))
summary(WSRules)
#people do not buy wine and soda together

#we find subset of rules that has Juice in LHS
JuiceRulesLHS <- subset(Rules_T, subset = lhs %ain% "Juice")
inspect(JuiceRulesLHS[1:5])
summary(JuiceRulesLHS)
plot(JuiceRulesLHS)

#we find subset of rules that has Juice in RHS
JuiceRulesRHS <- subset(Rules_T, subset = rhs %ain% "Juice")
inspect(JuiceRulesRHS[1:5])
summary(JuiceRulesRHS)
plot(JuiceRulesRHS)

#--------------Part 2----------------#

#Canned Vegetables & Fruits on the Right Hand Side
CannedRulesRHS <- subset(Rules_T, subset = rhs %pin% "Canned") 
summary(CannedRulesRHS)
inspect(sort(CannedRulesRHS, by = "lift")[1:5])
plot(CannedRulesRHS)

#Canned Vegetables & Fruits on the Left Hand Side
CannedRulesLHS <- subset(Rules_T, subset = lhs %pin% "Canned") 
summary(CannedRulesLHS)
inspect(sort(CannedRulesLHS, by = "lift")[1:5])
plot(CannedRulesLHS)

#Fresh Vegetables & Fruits on the Right Hand Side
FreshRulesRHS <- subset(Rules_T, subset = rhs %pin% "Fresh") 
summary(FreshRulesRHS)
inspect(sort(FreshRulesRHS, by = "lift")[1:5])
plot(FreshRulesRHS)

#Fresh Vegetables & Fruits on the Left Hand Side
FreshRulesLHS <- subset(Rules_T, subset = lhs %pin% "Fresh") 
summary(FreshRulesLHS)
inspect(sort(FreshRulesLHS, by = "lift")[1:5])
plot(FreshRulesLHS)

#we find association rules for 
#Fresh Vegetables/Fruits in LHS and Canned Vegetables/Fruits in RHS
FreshVsCanned <- subset(Rules_T, subset = lhs %pin% "Fresh" & rhs %pin% "Canned")
summary(FreshVsCanned)
inspect(sort(FreshVsCanned, by = "lift")[1:5])
plot(FreshVsCanned)

#we find association rules for 
#Fresh Vegetables/Fruits in RHS and Canned Vegetables/Fruits in LHS
CannedVsFresh <- subset(Rules_T, subset = rhs %pin% "Fresh" & lhs %pin% "Canned")
summary(CannedVsFresh)
inspect(sort(CannedVsFresh, by = "lift")[1:5])
plot(CannedVsFresh)

#create new rules of length two for finding association between Canned and Fresh 
CvFRules <- apriori(T_Data, parameter = list(support=0.01,conf=0.05, 
                                             minlen=2, maxlen=2))
summary(CvFRules)
inspect(head(CvFRules))
plot(CvFRules)

#we find association rules for 
#Fresh Vegetables/Fruits in RHS and Canned Vegetables/Fruits in LHS
CanVsFrsh <- subset(CvFRules, rhs %pin% 'Fresh' & lhs %pin% 'Canned')
summary(CanVsFrsh)
quality(CanVsFrsh)$kulc <- interestMeasure(CanVsFrsh, measure='kulczynski', significance=T, T_Data)
quality(CanVsFrsh)$kulc
inspect(CanVsFrsh)

#we find association rules for 
#Fresh Vegetables/Fruits in LHS and Canned Vegetables/Fruits in RHS
FrshVsCan <- subset(CvFRules, lhs %pin% 'Fresh' & rhs %pin% 'Canned')
summary(FrshVsCan)
quality(FrshVsCan)$kulc <- interestMeasure(FrshVsCan, measure='kulczynski', significance=T, T_Data)
quality(FrshVsCan)$kulc
inspect(FrshVsCan)

#-----------Part 3--------------#

#calculate the transaction sizes
T_Sizes <- as.numeric(levels(as.factor(size(T_Data))))
#list the transaction sizes
T_Sizes

#Small transaction rules
Small_T <- subset(T_Data, size(T_Data) <= 15)
summary(Small_T)
inspect(Small_T[1:25])

#Large transaction rules  
Large_T <- subset(T_Data, size(T_Data) > 15)
summary(Large_T)
inspect(Large_T[1:25])

#Transaction rules for size 1
One_T <- subset(T_Data, size(T_Data) == 1)
summary(One_T)
inspect(One_T[1:25])

#Transaction rules for size 44
FortyFour_T <- subset(T_Data, size(T_Data) == 44)
summary(FortyFour_T)
inspect(FortyFour_T)

#------------Part 4-------------#

#interesting association between Juice and Pancake Mix 
#it has high lift and kulc - suggesting positive corelation
Rules_P4 <- subset(Rules_T, subset = lhs %in% "Juice" & rhs %in% "Pancake Mix")
summary(Rules_P4)
inspect(sort(Rules_P4, by = "kulc")[1:25])
plot(Rules_P4)

#-------------End-----------------#