# Generate data distribution
dim=2
n_experts=3
n_one_record=200
n_modes=3
n_records=n_experts*n_one_record
mmd = MultiModalData(n_modes,dim,2)
df = mmd.generate(20000,[1 for i in range(n_modes)])

# Train local experts
df=[]
train_X=[]
train_y=[]
models=[] #experts
for i in range(n_experts):
    df1=mmd.generate(n_records/n_experts,random_distribution(n_modes))
    df.append(df1)
    train_X.append(df1[list(df1.columns[:-1])])
    train_y.append(df1['y'])
    expert1 = LocalSVM()
    expert1.fit(df1[list(df1.columns[:-1])], df1['y'])
    models.append(expert1)

# Define centralized meta model
flags = dict()
flags['dpsgd'] = False
flags['learning_rate'] = 0.15
flags['noise_multiplier'] = 1.1
flags['l2_norm_clip'] = 1.0
flags['batch_size'] = 250
flags['epochs'] = 300
flags['microbatches'] = 1
flags['model_dir'] = None

df_mm = mmd.generate(n_records,[1 for i in range(n_modes)])
meta_X_train, meta_y_train = df_mm[list(df1.columns[:-1])], df_mm['y']

# Meta model training data
X = np.reshape(np.array(meta_X_train), (n_experts,int(n_records/n_experts),-1))
Y = np.reshape(np.array(meta_y_train), (n_experts,int(n_records/n_experts),-1))

plot_data = mmd.generate(5000,[1 for i in range(n_modes)])
plot_X = plot_data[list(df1.columns[:-1])]
plot_y = plot_data['y']


for i in range(n_experts):
    print("Expert ",i, " ", models[i].accuracy(plot_X, plot_y))

# Joint training phase
krange=100
trange=1
acc_svm1=[]
acc_svm2=[]

meta_model_svm = MetaModel(input_shape=(dim,),  models=models, flags=flags)
for k in range(krange):
    meta_model_svm.update_gradient([meta_model_svm.compute_gradient(X[i], Y[i], update=False) for i in range(n_experts)])
    if k%5==0:
    print("num= ",k)
    val=meta_model_svm.accuracy(plot_X, plot_y)
    print(f"meta without DP : {val}")
    acc_svm1.append((k,val))

flags['dpsgd'] = True

meta_model_svm2 = MetaModel(input_shape=(dim,), models=models, flags=flags)
for k in range(krange):
    meta_model_svm2.update_gradient([meta_model_svm2.compute_gradient(X[i], Y[i], update=False) for i in range(n_experts)])
    if k%5==0:
    print("num= ",k)
    val=meta_model_svm2.accuracy(plot_X, plot_y)
    print(f"meta with DP : {val}")
    acc_svm2.append((k,val))


# First time, need to run visualization code below
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.plot(list(zip(*acc_svm1))[0], list(zip(*acc_svm1))[1], 'ro',label='Without DP')
plt.plot(list(zip(*acc_svm2))[0], list(zip(*acc_svm2))[1], 'bo',label='With DP')
for i in range(n_experts):
    e=models[i].accuracy(plot_X, plot_y)
    plt.plot([e for i in range(acc_svm1[-1][0])],label=f'expert{i}')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('epochs')
