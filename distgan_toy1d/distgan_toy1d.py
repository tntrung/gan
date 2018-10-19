import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.animation import ImageMagickWriter
from distgan import GAN

class Visualization(object):
    """Helper class to visualize the progress of the GAN training procedure.
    """
    def __init__(self, save_animation=False, fps=30):
        """Initialize the helper class.

        :param save_animation: Whether the animation should be saved as a gif. Requires the ImageMagick library.
        :param fps: The number of frames per second when saving the gif animation.
        """
        self.save_animation = save_animation
        self.fps = fps
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))
        self.figure.suptitle("1D GAAN")
        sns.set(color_codes=True, style='white', palette='colorblind')
        sns.despine(self.figure)
        plt.show(block=False)

        if self.save_animation:
            self.writer = ImageMagickWriter(fps=self.fps)
            self.writer.setup(self.figure, 'Toy_1D_GAAN.gif', dpi=100)

    def plot_progress(self, gan, session, data):
        """Plot the progress of the training procedure. This can be called back from the GAN fit method.

        :param gan: The GAN we are fitting.
        :param session: The current session of the GAN.
        :param data: The data object from which we are sampling the input data.
        """

        # Plot the training curve.
        steps = gan.log_interval * np.arange(len(gan.loss_d_curve))
        self.ax1.clear()
        self.ax1.plot(steps, gan.loss_d_curve, label='D loss')
        self.ax1.plot(steps, gan.loss_g_curve, label='G loss')
        
        self.ax1.set_title('Learning curve')
        self.ax1.set_xlabel('Iteration')

        if gan.model == 'gan' or gan.model == 'rgan' or gan.model == 'mdgan' or gan.model == 'vaegan' or gan.model == 'gaan':
            title = 'Loss'
        elif gan.model == 'wgangp':
            title = 'Negative critic loss'
        self.ax1.set_ylabel(title)

        # Plot the generated and the input data distributions.
        g = gan.sample(session)
        r1,r2 = self.ax2.get_xlim()
        x = np.linspace(r1, r2, gan.n_sample)[:, np.newaxis]

        critic = gan.dreal(session, x)

        if gan.model == 'wgangp':
            # Normalize the critic to be in [0, 1] to make visualization easier.
            critic = (critic - critic.min()) / (critic.max() - critic.min())
        d, _ = data.next_batch(gan.n_sample)
        if gan.model == 'gaan' or gan.model == 'rgan' or gan.model == 'mdgan':
            r    = gan.reconstruct(session, d)
            e    = gan.encode(session, d)
            
        self.ax2.clear()
        self.ax2.set_ylim([0, 1])
        self.ax2.set_xlim([-8, 8])
        if gan.model == 'gan' or gan.model == 'rgan' or gan.model == 'mdgan' or gan.model == 'gaan' or gan.model == 'vaegan':
            self.ax2.plot(x, critic, label='Decision boundary')
        elif gan.model == 'wgangp':
            self.ax2.plot(x, critic, label='Critic (normalized)')
        sns.kdeplot(d.flatten(), shade=True, ax=self.ax2, label='Real data')
       
        sns.kdeplot(g.flatten(), shade=True, ax=self.ax2, label='Generated data')
        self.ax2.set_title('Distributions')
        self.ax2.set_xlabel('Input domain')
        self.ax2.set_ylabel('Probability density')
        self.ax2.legend(loc='upper left', frameon=True)

        if len(steps) - 1 == gan.n_step // gan.log_interval:
            if self.save_animation:
                wait_seconds = 3
                [self.writer.grab_frame() for _ in range(wait_seconds * self.fps)]
                self.writer.finish()
            plt.show()
        else:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            if self.save_animation:
                self.writer.grab_frame()


class Dataset(object):
    """Dataset helper class which implements the next_batch method which can also be found in other TF helper classes.
    """
    def __init__(self, mu=4, sigma=1.0, seed=0):
        """Initialize the 1D Gaussian to generate.

        :param mu: Mean of the Gaussian.
        :param sigma: Standard deviation of the Gaussian.
        :param seed: Random seed to create a reproducible sequence.
        """
        self.seed = seed
        self.mu = mu
        self.sigma = sigma
        self.nmode = nmode
        np.random.seed(seed)

    def next_batch(self, batch_size):
        """Generate the next batch of toy input data.

        :param batch_size: Sample size.
        :return: Batch of toy data, generated from a 1D Gaussian.
        """
     	return np.random.normal(self.mu, self.sigma, (batch_size, 1)), np.ones(batch_size)


if __name__ == '__main__':
    seed   = 100  # Fix the random seed
    n_step = 10000  # Run the training procedure for 1450 steps.
    '''
    'gan'    : GANs
    'wgangp' : WGAN-GP
    'rgan'   : Regularized GANs
    'mdgan'  : MDGAN
    'vaegan' : VAEGAN
    'gaan'   : GAAN (or New name Dist-GAN)
    '''
    model = 'gaan'
    data  = Dataset(seed=seed)  # Initialize the toy data object.
    vis   = Visualization(save_animation=False)  # Create the visualization and optionally save the resulting animation.
    model = GAN(model=model, n_step=n_step, callback=vis.plot_progress, seed=seed).fit(data)  # Fit the data and plot the progress.
