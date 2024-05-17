import matplotlib.pyplot as plt


class VisualizeStrategy():

    def __init__(self):
        self.portfolio_values = []
        self.asset_allocation = []
        self.steps = []
        self.counter = 0

    def update_graph(self, portfolio_value, asset_allocation, assets):
        """
        Plot two graphs over time: portfolio value and pie chart with asset allocation
        """    
        self.portfolio_values.append(portfolio_value)
        self.asset_allocation = asset_allocation
        self.steps.append(self.counter)
        self.counter += 1

        plt.subplot(2, 1, 1)
        plt.title('Portfolio value over time')
        plt.xlabel('Step')
        plt.plot(self.steps, self.portfolio_values, color='blue')
        plt.ylabel('Price')


        plt.subplot(2, 1, 2)
        plt.title('Asset allocation')
        plt.pie(self.asset_allocation, labels=assets)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        plt.clf()
