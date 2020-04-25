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
        self.interaction_field=[]
        self.difussion_field=[]
        self.external_field=[]
        self.force_field=[]
        self.distance=None
        self.energy=None
        self.interaction_potential=0
        self.velocity_potential=0
        self.direction=None
