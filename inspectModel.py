import h2o
h2o.init()
model = h2o.load_model("DeepLearning_model_R_1678696295225_1")

print(model)

model.summary()

