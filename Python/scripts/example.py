from lib.UnsupervisedNN import UnsupervisedNN

items = [[1.0, 0.0]*3+[0.5, 0.5]*3+[0.0, 1.0*3]]
nn = UnsupervisedNN(2, 3, 0.1, 0.01)
for i in range(1000):
    for item in items:
        nn.train(item)

for item in items:
    cluster = nn.get_cluster(item)
    print(str(item) + " " + str(cluster))