

val_input_numpy = val_input  # Since val_input is already a numpy array

# Create a Gradient SHAP explainer
background = torch.zeros((1, *val_input_numpy.shape[1:]), device=device)
explainer = shap.GradientExplainer(model.to(device), background)

# Ensure model is on GPU
model.to(device)
model.train()  # set to training mode for SHAP

# Compute SHAP values
val_input_tensor = torch.tensor(val_input).float().to(device)

# Compute SHAP values
shap_values = explainer.shap_values(val_input_tensor,nsamples=2)

# After computing SHAP values, revert model to evaluation mode
model.eval()

with torch.no_grad():
      expected_value = np.array([model(background).mean().cpu().numpy()])

print(shap_values[0][0][0].flatten().shape)
print(val_input_numpy[0].flatten().shape)




import matplotlib.pyplot as plt
import shap

# Ensure that the SHAP values are in a numpy array format
if isinstance(shap_values, list):
    shap_values_matrix = np.array(shap_values[0])
else:
    shap_values_matrix = shap_values

# Check the shapes
print("Shape of shap_values:", shap_values_matrix.shape)
print("Shape of features (val_input_numpy):", val_input_numpy.shape)

# Plotting the summary plot
aggregated_shap_values = shap_values_matrix.sum(axis=1)
aggregated_features = val_input_numpy.sum(axis=1)

# Now, only 8 feature names are needed
simple_feature_names = ["(1)BIG", "(2)MID", "(3)SMA", "(4)BRAND", "train", "brancnt", "weekday", "prices"]

shap.initjs()
fig = plt.figure(figsize=(12, 12))
fig.set_facecolor('white')
ax = fig.add_subplot()



shap.summary_plot(aggregated_shap_values, aggregated_features, feature_names=simple_feature_names,plot_type="dot",show=False)


plt.savefig("/data/user/postreview/output/summary_plot_2withall.png")

