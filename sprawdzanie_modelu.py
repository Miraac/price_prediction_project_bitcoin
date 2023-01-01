from sklearn.model_selection import train_test_split

# Podziel dane na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Trenuj model na zestawie treningowym
model.fit(X_train, y_train, epochs=10)

# Oceń model na zestawie testowym
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
