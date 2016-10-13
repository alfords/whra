# Map 1-based optional input ports to variables

train = maml.mapInputPort(1) # class: data.frame

require(xgboost)
require(data.table)
require(Matrix)

# imputation to replace all missing values by -9999
train[is.na(train)] = -9999

# create a data.table format for the train data
train.dt=data.table(train)

# create the unique labels from geo, segment and subgroup
unique_label = train.dt[,unique(100*geo+10*segment+subgroup)]
# number of unique classes
mclass=length(unique_label)

# since the predicted classes from xgb is 0..(mclass-1), mapping between unique_label and predicted classes is needed.
label_mapping=data.table(unique_label=c(111, 121, 122 ,211 ,212 ,221 ,222, 231, 241 ,311, 312, 321 ,322, 411, 412, 511 ,512 ,522, 531, 532, 611, 612 ,621, 711,712, 721, 722, 731, 811 ,821 ,831, 841, 911, 912,921, 931 ,932),
  order=c(26 ,19 , 9, 28, 36 , 3 ,16 , 2 ,25,  4 ,31, 34, 20 , 1 ,21 ,18 ,17 ,10, 13, 33  ,7 ,12, 11, 35, 22 ,32, 14 ,23 ,15 , 6 , 5 ,29 , 8,27 ,24 , 0, 30))
setkey(label_mapping,'unique_label')

# add the label/class column in the train data
train.dt[,unique_label:=100*geo+10*segment+subgroup]
setkey(train.dt,'unique_label')

# join the train data and mapping table to create a new column of label/class in the train data
train.dt=train.dt[label_mapping]
train.dt$unique_label=NULL

# how many features in train data
ncols = ncol(train)

# select all the available features
feature_index= c(2:18, 20:(ncols-2))

########## deal with character feature religion by converting the religion column to integer #########
rel_map=data.table(rel=sort(unique(train.dt$religion)),rel_num=c(1:length(unique(train.dt$religion))))

setkey(rel_map,'rel')
print(rel_map)
setkey(train.dt,'religion')

head(train.dt,10)
train.dt=train.dt[rel_map]
head(train.dt,10)

train.dt[,religion:=rel_num]

train.dt$rel_num=NULL

train_y=train.dt$order
train.dt$order=NULL

# train data ready to use
print('data prepared')

# convert the data from data.table to data.frame 
train.df=data.frame(train.dt)

summary(train.df[,feature_index])

# 
set.seed(2016)

dtrain=xgb.DMatrix(data=data.matrix(train.df[,feature_index])+0.0,label=train_y,missing=-9999)

print('start xgb')

# set the parameters for xgboost
params=list(
  booster='gbtree',
  objective='multi:softmax',
  lambda=15,
  subsample=0.8,
  colsample_bytree=0.75,
  min_child_weight=2,
  max_depth=8,
  eta=0.06,
  eval_metric='merror',
  num_class=37
)
# number of rounds to use
nrounds=360
model_xgb=xgb.train(params = params,data=dtrain,nrounds = nrounds)
print('prediction finished!')

# save the xgb model
raw_model = xgb.save.raw(model_xgb)
data.set = data.frame(model=as.integer(raw_model))
maml.mapOutputPort("data.set");
