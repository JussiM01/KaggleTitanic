import matplotlib
import matplotlib.pyplot as predictions_file_object

def pl(data, x, y):
  ax = data[data['Survived'] == 0].plot(x = x, y = y, kind = 'scatter', c = 'red')
  data[data['Survived'] == 1].plot(x = x, y = y, kind = 'scatter', c = 'blue', ax = ax)
  plt.show()
