import numpy as np


class StressTensor:
    
    def __init__(self,
                sx: float = 0,
                sy: float = 0,
                sz: float = 0,
                sxy: float = 0,
                syz: float = 0,
                szx: float = 0,
                ) -> None:
        
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.sxy = sxy
        self.syz = syz
        self.szx = szx
    
    @property
    def stressComponents(self) -> list:
        """Returns six stress components in list form.
        Order: sx, sy, sz, sxy, syz, zx

        """
        return [self.sx, self.sy, self.sz, self.sxy, self.syz, self.szx]
    
    @property
    def stressTensor(self) -> list:
        """Returns stress tensor matrix:
        |sxx sxy sxz|
        |sxy syy syz|
        |sxz syz szz|
        
        """
        return [[self.sx, self.sxy, self.szx],
                [self.sxy, self.sy, self.syz],
                [self.szx, self.syz, self.sz]]

    @property
    def principals(self) -> list:
        return np.linalg.eigvals(self.stressTensor)
    
    @property
    def firstPrincipal(self) -> float:
        return max(self.principals)
    
    @property
    def secondPrincipal(self) -> float:
        return self.principals[1]
    
    @property
    def firstPrincipal(self) -> float:
        return min(self.principals)
    
    @property
    def eigenVector(self) -> list:
        val, vect = np.linalg.eigvals(self.stressTensor)
        return vect
    
    @property
    def HuberStress(self) -> float:
        sx_diff = (self.sx - self.sy)**2
        sy_diff = (self.sy - self.sz)**2
        sz_diff = (self.sz - self.sx)**2
        return np.sqrt(0.5*(sx_diff+sy_diff+sz_diff) + 3*(self.sxy**2 +self.syz**2 +self.szx**2))
    
    def applyComponentKt(self,
                         KTx: float = 1,
                         KTy: float = 1,
                         KTz: float = 1,
                         KTxy: float = 1,
                         KTyz: float = 1,
                         KTZX: float = 1) -> None:
        """Allows to apply component Kt

        Args:
            KTx (float, optional): Scaling factor for sxx. Defaults to 1.
            KTy (float, optional): Scaling factor for syy. Defaults to 1.
            KTz (float, optional): Scaling factor for szz. Defaults to 1.
            KTxy (float, optional): Scaling factor for sxy. Defaults to 1.
            KTyz (float, optional): Scaling factor for syz. Defaults to 1.
            KTZX (float, optional): Scaling factor for szx. Defaults to 1.
        """
        self.sx  = self.sx  *KTx
        self.sy  = self.sy  *KTy
        self.sz  = self.sz  *KTz
        self.sxy = self.sxy *KTxy
        self.syz = self.syz *KTyz
        self.szx = self.szx *KTZX
        
    def calculateWalker(self, tensor) -> float:
        
        sm_comp = [(cmp1+cmp2)/2 for cmp1, cmp2 in zip(self.stressComponents, tensor.stressComponents)]
        sa_comp = [(cmp1-cmp2)/2 for cmp1, cmp2 in zip(self.stressComponents, tensor.stressComponents)]
        
        s_mx, s_my, s_mz, s_mxy, s_myz, s_mzx = sm_comp
        s_ax, s_ay, s_az, s_axy, s_ayz, s_azx = sa_comp
        
        sm = np.sign([s_mx, s_my, s_mz]) *np.sqrt(2)/2*self._calcMises(s_mx, s_my, s_mz, s_mxy, s_myz, s_mzx)
        sa = np.sign([s_ax, s_ay, s_az]) *np.sqrt(2)/2*self._calcMises(s_ax, s_ay, s_az, s_axy, s_ayz, s_azx)
        
        return sa
        
    
    @staticmethod
    def _calcMises(x: float, y: float, z: float, xy:float, yz:float, zx:float) -> float:
        return np.sqrt((x-y)**2+(y-z)**2+(z-x)**2 +6*(xy**2 +yz**2 +zx**2))
        
        
        
    def __add__(self, cls):
        new_sx  = self.sx  + cls.sx 
        new_sy  = self.sy  + cls.sy 
        new_sz  = self.sz  + cls.sz 
        new_sxy = self.sxy + cls.sxy
        new_syz = self.syz + cls.syz
        new_szx = self.szx + cls.szx
        
        return StressTensor(sx = new_sx,
                            sy = new_sy,
                            sz = new_sz,
                            sxy = new_sxy,
                            syz = new_syz,
                            szx = new_szx)
        
    def __eq__(self, value: object) -> bool:
        """Compares two stress tensors. If all six components are same returns True.

        Args:
            value (np.object): _description_

        Returns:
            bool: _description_
        """
        if (self.sx == value.sx and
            self.sy == value.sy and
            self.sz == value.sz and
            self.sxy == value.sxy and
            self.syz == value.syz and
            self.szx == value.szx):
            
            return True
        
        else:
            return False

