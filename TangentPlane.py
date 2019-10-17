class tangent_plane():
    def __init__(self):
        self.particles=[]
        self.divergence=[]
        self.metric=[]
        self.density=[]
        self.num_particles= 0
        self.gradient=[]
        self.reg_density=[]
        self.ball=None
        self.interaction_field=0
        self.distance=None
        self.energy=None
