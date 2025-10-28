from MoeaBench.base_benchmark import BaseBenchmark

class my_dtlz5(BaseBenchmark):

    from enum import Enum
    import numpy as np


    class E_DTLZ(Enum):
       F1   = 1
       F2   = 2
       F3   = 3
       Fm   = 5


    def __init__(self,CACHE,M=3,P=150,K=5,n_ieq_constr=1):
        self.M=M
        self.P=P
        self.K=K
        self.n_ieq_constr=n_ieq_constr
        self.llist_E_DTLZ = list(self.E_DTLZ)
        self.N=self.K+self.M-1
        self.CACHE=CACHE


    def get_CACHE(self):
       return self.CACHE


    def constraits(self,f,parameter = 1,f_c=[]):
        f_constraits=self.np.array(f)
        f_c = self.np.array([self.np.sum([ f_c**2  for  f_c in f_constraits[linha,0:f_constraits.shape[1]]])-parameter for index,linha in enumerate(range(f_constraits.shape[0]))  ])
        return f_c


    def eval_cons(self,f):
        const_in=[]
        M_constraits = self.constraits(f)
        for (fc,fo) in zip(M_constraits,f):
            if float(fc) == 0:
                const_in.append(fo)
        return self.np.array(const_in)


    def get_Points(self):
        return self.np.array([*self.np.random.random((self.P, self.N))*1.0])


    def F1(self,M,th,Gxm):
       theta = list(map(lambda TH: self.np.cos(TH), th[0:(M-1)]))
       return (1+Gxm)*self.np.prod(self.np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)


    def F2(self,M,th,Gxm):
        theta = list(map(lambda TH: self.np.cos(TH), th[0:(M-2)]))
        return (1+Gxm)*self.np.prod(self.np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)*self.np.column_stack(self.np.sin(th[(M-2):(M-1)]))


    def F3(self,M,th,Gxm):
        theta = list(map(lambda TH: np.cos(TH), th[0:(M-3)]))
        return (1+Gxm)*self.np.prod(self.np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)*self.np.column_stack(self.np.sin(th[(M-3):(M-2)]))


    def Fm(self,M,th,Gxm):
        return (1+Gxm)*self.np.column_stack(self.np.sin(th[0:1]))


    def get_method(self,enum):
        return self.llist_E_DTLZ[enum]


    def param_F(self):
        dict_F = {
                    self.get_method(0) : self.F1,
                    self.get_method(1) : self.F2,
                    self.get_method(2) : self.F3,
                    self.get_method(3) : self.Fm
                  }
        return dict_F


    def calc_F_M(self,Fi,M):
        if Fi == 1:
            return self.get_method(0)
        elif Fi == 2 and M > 2:
            return self.get_method(1)
        elif Fi >= 3 and Fi <= M-1 and M > 3:
            return self.get_method(2)
        elif Fi == M:
            return self.get_method(3)


    def calc_TH(self,X,Gxm,M):
        return [X[:,Xi:Xi+1]*self.np.pi/2 if Xi == 0 else (self.np.pi/(4*(1+Gxm))*(1+2*Gxm*X[:,Xi:Xi+1]))  for Xi in range(0,M-1)]


    def calc_f(self,X,G):
        vet_F_M = [self.calc_F_M(F,self.M) for F, i in enumerate(range(0,self.M), start = 1)]
        return self.np.column_stack(list(map(lambda Key: self.param_F()[Key](self.M,self.calc_TH(X,G,self.M),G),vet_F_M)))


    def calc_g(self,X):
        return self.np.sum((X[:,self.M-1:]-0.5)**2, axis = 1).reshape(X.shape[0],1)


    def POFsamples(self):
        X = self.get_Points()
        X[:,self.M-1:self.N]=0.5
        G = self.calc_g(X)
        F = self.eval_cons(self.calc_f(X,G))
        return F


    def evaluation(self,x,n_ieq):
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F}
        if n_ieq != 0:
            cons = self.constraits(F,1.25)
            const  = cons.reshape(cons.shape[0],1)
            result["G"] = const
            result["feasible"] = self.np.any((result["G"] <-0.00000000001)  | (result["G"] > 0.00000000001) )
        return result