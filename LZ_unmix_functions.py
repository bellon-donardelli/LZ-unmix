import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as integrate
import pandas as pd


plt.rcParams.update({'font.size': 14})


def f1(X0,Bc,W,A,x):
    
    '''
    A direct model of the  the first derivative of the hysteresis loop, which is a phenomenological model based on the Lorentzian function.
    Here, there is only one ferromagnetic component (Ca).
    
    Parameters: 
        X0 : a float, the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc : a float, the coercive force, where ferromagnetic contribution is zero;
        W : a float, estimated as the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A : a float, a value corresponding to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        
    returns: 
        y: a 1D-array of values.
        '''
    const=(2*A)/np.pi
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=(abs(W)/((4*((x[i]-abs(Bc))**2))+(W**2))*const)+X0

    return y




def f2(X0,Bc1,W1,A1,Bc2,W2,A2,x):
    '''
    A direct model of the the first derivative of the hysteresis loop, which is a phenomenological model based on the Lorentzian function.
    Here, there are two ferromagnetic components (Ca, Cb).
    
    Parameters: 
        X0 : a float, the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1\Bc2 : a float, the coercive force, where ferromagnetic contribution is zero;
        W1\W2 : a float, estimated as the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1\A2 : a float, a value corresponding to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        
    returns:  
        Y: a 1D-array of values.

    '''
    const1=X0
    const2=2/np.pi
    Y=np.zeros(np.size(x))
    for i in range(np.size(Y)):
        Y[i]=(const2*((abs(A1)*(abs(W1)/((4*((x[i]-abs(Bc1))**2))+(W1**2))))+(abs(A2)*(abs(W2)/((4*((x[i]-abs(Bc2))**2))+(W2**2))))))+const1

    return Y
    


def f3(X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3,A3,x):
    '''
    A direct model of the the first derivative of the hysteresis loop, which is a phenomenological model based on the Lorentzian function.
    Here, there are three ferromagnetic components (Ca, Cb, Cc).
    
    Parameters: 
        X0 : a float, the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1\Bc2\Bc3 : a float, the coercive force, where ferromagnetic contribution is zero;
        W1\W2\W3 : a float, estimated as the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1\A2\A3 : a float, a value corresponding to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        
    returns:  
        Y: a 1D-array of values.
    '''
    const1=X0
    const2=2/np.pi
    Y=np.zeros(np.size(x))
    for i in range(np.size(Y)):
        Y[i]=(const2*((abs(A1)*(abs(W1)/((4*((x[i]-abs(Bc1))**2))+(W1**2))))+(abs(A2)*(abs(W2)/((4*((x[i]-abs(Bc2))**2))+(W2**2))))+
            (abs(A3)*(abs(W3)/((4*((x[i]-abs(Bc3))**2))+(W3**2))))))+const1

    return Y



def Ms(A):
    '''
    Calculates the magnetization saturation (Ms) of a given ferromagnetic component from the phenomenological model based on the Lorentzian function.

    Parameters:
        A: a float, a value corresponding to integral under the Lorentzian component;

    Returns:
        Ms: a float. 
    '''
    Ms=A/2
    return Ms

def Mrs(Bc,W,A):
    '''
    Calculates the magnetization saturation of remanence (Mrs) of a given ferromagnetic component from the phenomenological model based on the Lorentzian function.

    Parameters:
        Bc : a float, the coercive force, where ferromagnetic contribution is zero;
        W: a float, estimated as the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A : a float, a value corresponding to integral under the Lorentzian component;

    Returns:
        Mrs: a float. 
    '''
    Mrs=(A/np.pi)*np.arctan((-2*Bc)/W)
    Mrs=abs(Mrs)
    return Mrs




def gradient(x,y,pars='yes'):
    
    '''
    Calculates the ∂y/∂x through a finite-differences method.
    
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        y : a 1D-array containing the values of the dependent variable
        pars: string, 'yes' or 'no', which indicates if the border values will be ignored.
        
    returns:
        grad: a 1D-array of values. 
        '''
        
    grad=np.zeros(np.size(x))
    for i in range(np.size(x)):
        if i==0:
            grad[i]=np.array((y[i+1]-y[i])/(x[i+1]-x[i]))
        if i>0 and i<(np.size(x)-1):
            grad[i]=np.array((y[i+1]-y[i-1])/(x[i+1]-x[i-1]))
        else:
            grad[i]=np.array((y[i]-y[i-1])/(x[i]-x[i-1]))
    
    if pars=='yes' or pars=='y' or pars=='YES' or pars =='Yes':
        grad[-1]=grad[-2]
        grad[0]=grad[1]
    elif pars=='no' or pars=='n' or pars=='NO' or pars =='No':
        grad=grad
    return(grad)


def numerical_int(data,xzero,xf):
    '''
    Calculates the an aproximation to a numerical integration, the defined area under a curve.
    
    Parameters: 
        data : a 1D-array containing the values of the curve;
        xzero : an integer, the left interval delimeter;
        xf: an integer, the right interval delimeter.
        
    returns: 
        integral: a float. 
        '''
    integral=(xf-xzero)*np.mean(data)
    
    return integral


def region_selection(x,y,yi,yf):
    '''
    Sample the data (x,y) in a given interval.
    
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        y : a 1D-array containing the values of the dependent variable
        yi: a integer, the left interval delimeter
        yf: a integer, the right interval delimiter
        
    returns:
        x_new: a 1D-array of values;
        y_new: a 1D-array of values.
    '''

    y_new=[]
    x_new=[]
    for i in range(np.size(y)):
        if i>=yi and i<=yf:
            y_new=np.append(y_new,y[i])
            x_new=np.append(x_new,x[i])
    
    return x_new,y_new


def region_selection2(y,yi,yf):
    '''
    Sample the data (y) in a given interval.
    
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        yi: a integer, the left interval delimeter
        yf: a integer, the right interval delimiter
        
    returns:
        x_new: a 1D-array of values.
    '''
    y_new=[]
    for i in range(np.size(y)):
        if i>=yi and i<=yf:
            y_new=np.append(y_new,y[i])

    return y_new
    

def line_inv(x,y):
    '''
    Perform a linear regression (y=ax+b) through a least squares matrix methodology.
    Parameters: 
        x : a 1D-array containing the values of the independent variable
        y : a 1D-array containing the values of the dependent variable

    returns:
        a: a float, the slope;
        b: a float, the linear coefficient.

    '''
    gradb=np.ones(np.size(x))
    G=np.column_stack([x,gradb])
    a,b=(np.linalg.inv(G.T@G))@G.T@y
    

    return a,b



def ferro_area(x,data,selecting_function,lineinv_function,numerical_int):
    '''
    Firstly subtracts the high-field para/dia susceptibility, then calculates the total area corresponding to the ferromagnetic components.

    Parameters: 
        x : a 1D-array containing the values of the independent variable;
        data : a 1D-array containing the values of the dependent variable;
        selecting_function: a function that will sample the high field region, region_selection;
        lineinv_function: a function that provides linear regression parameters (y=ax+b), line_inv;
        numerical_int: a function that calculate numerical integrals, numerical_int.


    returns:
        Area: a float.

    '''
    xs1,ys1=selecting_function(x,data,0,int(np.size(data)/18)) #constraining the values at high fields to invert a line representing para/dia contribution

    xs2,ys2=selecting_function(x,data,(int(np.size(data)/2)+int(np.size(data)/2.2)),int(np.size(data)))#constraining the values at high fields to invert a line representing para/dia contribution

    x3,y3=np.concatenate((xs1,xs2)),np.concatenate((ys1,ys2)) #concatening new section to invert a line at high-fields
    
    a,b=lineinv_function(x3,y3) #invertion of a line to find the slope
    y_line=(a*x)+b #calculating the line inverted for high-field values
    
    Area=numerical_int(data-y_line,x[0],x[-1]) #calculating the Area under the smoothed curve

    Area=np.array([Area])
  
    return Area




def moving_mean(x,group_size=30):

    '''
    Computes a moving mean filter on a given interval.

    Parameters: 
        x : a 1D-array containing the values of the independent variable;
        group_size: an integer >=1, delimiting the number of points;


    returns:
        moving_meanx: a 1D-array of values.

    '''
    i = 0
    moving_meanx=[]
    
    # Calculate the means:
    while i < len(x) - group_size  + 1:
        group = x[i : i + group_size ]
        mean_group = sum(group) / group_size 
        moving_meanx.append(mean_group)
        i +=1
    

    moving_meanx=np.array(moving_meanx)

    return moving_meanx



def Levenberg_Marquardt1(function,X0,Bc1,W1,A1,x,data,constrain,condition=1e-5,maxiter=100,index=' sample',sx=2.0E-5,sy=10**-12):
    '''
    Optimization of initial guesses (X0,Bc1,W1 and A1),through a Levenberg Marquardt method. As this function plots every single iteration, it
    should be used only to the final model. 

    Parameters: 
        function : phenomenological model based on the Lorentzian function;
        X0 : initial guess (float) for the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1 : initial guess (float) for the coercive force, where ferromagnetic contribution is zero;
        W1: initial guess (float) for the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1: initial guess (float) for to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        data : a 1D-array containing the values of the dependent variable (lower branch of a magnetic hysteresis)
        constrain: a string, "Y/N", which will determinate if Bc is fixed or optimized.
        condition: a float, being a small value used both in the calculation of the numerical derivatives as a way of residum comparison.
        maxiter: an integer, representing the maximum number of iterations calculated through an inversion process (if the condition is not reach).
        index: a string, the name of the sample. Defaut(' sample').
        sx: a float, the instrumental uncertainty of the field measurements. Defaut (2.0E-5T).
        sy: a float, the instrumental uncertainty of the magnetic moment measurements. Defaut (10**-12 Am²).


    returns: 
        m_zero: a 1D-array with the optimized parameters(X0,Bc1,W1 and A1);
        deltad_f:a float, the euclidean norm of the error;
        uncertainty: a 1D-array with the calculated uncertainty at each (x,y) point;
        y_iteration: a 1D-array, a direct model calculated through the optimized parameters (the inverted model).
    

    '''
    a=0.2
    b=1
    damping=1
    itr=0
    bubble_deltad=[]
    bubble_damping=[]
    bubble_damping=[]
    m_zero=np.array([X0,Bc1,W1,A1])
    m_cor=np.zeros(np.shape(m_zero))

    if constrain=='N':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],x))/(2*condition)
                    grad_Bc1=(function(m_zero[0],m_zero[1]+condition,m_zero[2],m_zero[3],x)-function(m_zero[0],m_zero[1]-condition,m_zero[2],m_zero[3],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,x))/(2*condition)
            
                    J=np.column_stack([grad_yzero,grad_Bc1,grad_W1,grad_A1])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1]+par_correct[1],
                                m_zero[2]+par_correct[2],m_zero[3]+par_correct[3]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                    c1=function(0,m_zero[1],m_zero[2],m_zero[3],x)
                    itr+=1
                    figure,ax=plt.subplots(figsize=(5,5))
                    ax.scatter(x,data,alpha=0.2,color='gray',label=r'$Data$')
                    ax.plot(x,y_iteration,color='k',label=r'$C_{a}inv$')
                    ax.plot(x,c1,label=r'$C_{a}$',color='royalblue')
                    ax.legend(loc='best',shadow=True)
                    ax.grid(alpha=0.5)
                    ax.set_xlabel(r'$B \ (T)$')
                    ax.set_ylabel(r'$\partial{M} / \partial{B} $')
                    ax.set_title(f'$Loop: \ {itr +1} $')
                    figure.tight_layout()
                    plt.savefig('Inversion_routine '+str(index)+'.pdf',dpi=300,facecolor='w')


                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   
    if constrain=='Y':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,x))/(2*condition)
            
                    J=np.column_stack([grad_yzero,grad_W1,grad_A1])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1],
                                m_zero[2]+par_correct[1],m_zero[3]+par_correct[1]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                    c1=function(0,m_zero[1],m_zero[2],m_zero[3],x)
                    itr+=1
                    figure,ax=plt.subplots(figsize=(5,5))
                    ax.scatter(x,data,alpha=0.2,color='gray',label=r'$Data$')
                    ax.plot(x,y_iteration,color='k',label=r'$C_{a}inv$')
                    ax.plot(x,c1,label=r'$C_{a}$',color='royalblue')
                    ax.legend(loc='best',shadow=True)
                    ax.grid(alpha=0.5)
                    ax.set_xlabel(r'$B \ (T)$')
                    ax.set_ylabel(r'$\partial{M} / \partial{B} $')
                    ax.set_title(f'$Loop: \ {itr +1} $')
                    figure.tight_layout()
                    plt.savefig('Inversion_routine '+str(index)+'.pdf',dpi=300,facecolor='w')


                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   


    XY=np.stack((x,data), axis=0)
    COV_XY=np.cov(XY)
    sigma_squared_y=(sy)**2
    sigma_squared_x=(sx)**2
    COV_P=np.linalg.inv((J.T@J))*sigma_squared_y
    gradient_x=np.gradient(x)
    gradient_x[-1]=gradient_x[-2]
    gradient_x[0]=gradient_x[1]
    uncertainty=[]

    if constrain=='N':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((gradient_x[i]**2)*sigma_squared_x)))

        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])

    if constrain=='Y':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((gradient_x[i]**2)*sigma_squared_x)))

        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])



    figure,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3))
    ax1.plot(bubble_deltad,marker='.',color='gray')
    ax1.set_ylabel('||e||²')
    ax1.set_xlabel('Step')
    ax1.grid(alpha=0.5)
    ax2.plot(bubble_damping,marker='.',color='m')
    ax2.set_ylabel('λ')
    ax2.set_xlabel('Step')
    ax2.grid(alpha=0.5)
    figure.tight_layout()
    m_zero[1]=abs(m_zero[1])
    m_zero[2]=abs(m_zero[2])
    m_zero[3]=abs(m_zero[3])
    
    plt.savefig('Optmization_parameters '+str(index)+'.pdf',dpi=300,facecolor='w')
    
    if constrain=='N':
        df=pd.DataFrame(data={'y0':m_zero[0],
                          'Bc1':m_zero[1],
                          'W1':m_zero[2],
                          'A1':m_zero[3],
                          'var_yzero':COV_P[0,0],
                          'var_Bc1':COV_P[1,1],
                          'var_W1':COV_P[2,2],
                          'var_A1':COV_P[3,3]},index=[0])

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)

        df1=pd.DataFrame(data={'x1':x,
                          'Ca':c1,
                          'Ca_total':y_iteration,
                          'uncertainty':uncertainty,
                          'gradient':data})

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)
        df1.to_csv('inversion_components_'+str(index)+'.csv', index=False)

    if constrain=='Y':
        df=pd.DataFrame(data={'y0':m_zero[0],
                          'Bc1':m_zero[1],
                          'W1':m_zero[2],
                          'A1':m_zero[3],
                          'var_yzero':COV_P[0,0],
                          'var_W1':COV_P[1,1],
                          'var_A1':COV_P[2,2]},index=[0])

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)

        df1=pd.DataFrame(data={'x1':x,
                          'Ca':c1,
                          'Ca_total':y_iteration,
                          'uncertainty':uncertainty,
                          'gradient':data})

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)
        df1.to_csv('inversion_components_'+str(index)+'.csv', index=False)

    figure,ax3=plt.subplots(figsize=(5,5))
    ax3.plot(x,y_iteration,color='r',zorder=20)
    ax3.plot(x,data,color='gray',marker='.',label='Data',alpha=0.3)
    ax3.fill_between(x,y_iteration+uncertainty,y_iteration-uncertainty,alpha=0.4,color='royalblue',label='Uncertainty',zorder=20)
    ax3.set_xlabel(r'$B \ (T)$')
    ax3.set_ylabel(r'$\partial{M} / \partial{B} $')
    ax3.grid(alpha=0.5)
    ax3.legend(shadow=True,loc='upper right')
    ax3.set_ylim(min(data)-(min(data)*0.1),(max(data)*0.1)+max(data))
    figure.tight_layout()

    plt.savefig('Final_inversion_model'+str(index)+'.pdf',dpi=300,facecolor='w')

    deltad_f=np.linalg.norm(delta_d,2)
    return m_zero,deltad_f,uncertainty,y_iteration


def Levenberg_Marquardt2(function,X0,Bc1,W1,A1,Bc2,W2,A2,x,data,function2,constrain,condition=1e-5,maxiter=100,index=' sample',sx=2.0E-5,sy=10**-12):
    '''
    Optimization of initial guesses (X0,Bc1,W1,A1,Bc2,W2 and A2),through a Levenberg Marquardt method.As this function plots every single iteration, it
    should be used only to the final model. 

    Parameters: 
        function : phenomenological model based on the Lorentzian function;
        X0 : initial guess (float) for the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1\Bc2 : initial guess (float) for the coercive force, where ferromagnetic contribution is zero;
        W1\W2: initial guess (float) for the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1\A2: initial guess (float) for to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        data : a 1D-array containing the values of the dependent variable (lower branch of a magnetic hysteresis)
        constrain: a string, "Y/N", which will determinate if Bc is fixed or optimized.
        condition: a float, being a small value used both in the calculation of the numerical derivatives as a way of residum comparison.
        maxiter: an integer, representing the maximum number of iterations calculated through an inversion process (if the condition is not reach).
        index: a string, the name of the sample. Defaut(' sample').
        sx: a float, the instrumental uncertainty of the field measurements. Defaut (2.0E-5T).
        sy: a float, the instrumental uncertainty of the magnetic moment measurements. Defaut (10**-12 Am²).


    returns: 
        m_zero: a 1D-array with the optimized parameters(X0,Bc1,W1,A1,Bc2,W2 and A2);
        deltad_f:a float, the euclidean norm of the error;
        uncertainty: a 1D-array with the calculated uncertainty at each (x,y) point;
        y_iteration: a 1D-array, a direct model calculated through the optimized parameters (the inverted model).
    


    '''
    a=0.2
    b=1
    damping=1
    itr=0
    bubble_deltad=[]
    bubble_damping=[]
    bubble_damping=[]
    m_zero=np.array([X0,Bc1,W1,A1,Bc2,W2,A2])
    m_cor=np.zeros(np.shape(m_zero))
    if constrain=='N':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_Bc1=(function(m_zero[0],m_zero[1]+condition,m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1]-condition,m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_Bc2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4]+condition,m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4]-condition,m_zero[5],m_zero[6],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_Bc1,grad_W1,grad_A1,grad_Bc2,grad_W2,grad_A2])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1]+par_correct[1],
                                m_zero[2]+par_correct[2],m_zero[3]+par_correct[3],
                                m_zero[4]+par_correct[4],m_zero[5]+par_correct[5],m_zero[6]+par_correct[6]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    itr+=1
                    figure,ax=plt.subplots(figsize=(5,5))
                    ax.scatter(x,data,alpha=0.2,color='gray',label=r'$Data$')
                    ax.plot(x,y_iteration,color='k',label=r'$C_{a}+C_{b}$')
                    ax.plot(x,c1,label=r'$C_{a}$',color='royalblue')
                    ax.plot(x,c2,label=r'$C_{b}$',color='forestgreen')
                    ax.legend(loc='best',shadow=True)
                    ax.grid(alpha=0.5)
                    ax.set_xlabel(r'$B \ (T)$')
                    ax.set_ylabel(r'$\partial{M} / \partial{B} $')
                    ax.set_title(f'$Loop: \ {itr +1} $')
                    figure.tight_layout()
                    plt.savefig('Inversion_routine '+str(index)+'.pdf',dpi=300,facecolor='w')


                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   
    
    if constrain=='Y':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_W1,grad_A1,grad_W2,grad_A2])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1],
                                m_zero[2]+par_correct[1],m_zero[3]+par_correct[2],
                                m_zero[4],m_zero[5]+par_correct[3],m_zero[6]+par_correct[4]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],m_zero[5],m_zero[6],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    itr+=1
                    figure,ax=plt.subplots(figsize=(5,5))
                    ax.scatter(x,data,alpha=0.2,color='gray',label=r'$Data$')
                    ax.plot(x,y_iteration,color='k',label=r'$C_{a}+C_{b}$')
                    ax.plot(x,c1,label=r'$C_{a}$',color='royalblue')
                    ax.plot(x,c2,label=r'$C_{b}$',color='forestgreen')
                    ax.legend(loc='best',shadow=True)
                    ax.grid(alpha=0.5)
                    ax.set_xlabel(r'$B \ (T)$')
                    ax.set_ylabel(r'$\partial{M} / \partial{B} $')
                    ax.set_title(f'$Loop: \ {itr +1} $')
                    figure.tight_layout()
                    plt.savefig('Inversion_routine '+str(index)+'.pdf',dpi=300,facecolor='w')


                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   
    XY=np.stack((x,data), axis=0)
    COV_XY=np.cov(XY)
    sigma_squared_y=(sy)**2
    sigma_squared_x=(sx)**2
    COV_P=np.linalg.inv((J.T@J))*sigma_squared_y
    gradient_x=np.gradient(x)
    gradient_x[-1]=gradient_x[-2]
    gradient_x[0]=gradient_x[1]
    uncertainty=[]

    if constrain=='N':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,(np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
                        ((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+
                                                ((J[i,5]**2)*COV_P[5,5])+((J[i,6]**2)*COV_P[6,6])+((gradient_x[i]**2)*sigma_squared_x))))
        

        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])

    if constrain=='Y':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+
                        ((J[i,1]**2)*COV_P[1,1])+((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+((gradient_x[i]**2)*sigma_squared_x)))
        
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])

    figure,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3))
    ax1.plot(bubble_deltad,marker='.',color='gray')
    ax1.set_ylabel('||e||²')
    ax1.set_xlabel('Step')
    ax1.grid(alpha=0.5)
    ax2.plot(bubble_damping,marker='.',color='m')
    ax2.set_ylabel('λ')
    ax2.set_xlabel('Step')
    ax2.grid(alpha=0.5)
    figure.tight_layout()

    
    plt.savefig('Optmization_parameters '+str(index)+'.pdf',dpi=300,facecolor='w')

    if constrain=='N':
        df=pd.DataFrame(data={'y0':m_zero[0],
                          'Bc1':m_zero[1],
                          'W1':m_zero[2],
                          'A1':m_zero[3],
                          'Bc2':m_zero[4],
                          'W2':m_zero[5],
                          'A2':m_zero[6],
                          'var_yzero':COV_P[0,0],
                          'var_Bc1':COV_P[1,1],
                          'var_W1':COV_P[2,2],
                          'var_A1':COV_P[3,3],
                          'var_Bc2':COV_P[4,4],
                          'var_W2':COV_P[5,5],
                          'var_A2':COV_P[6,6]},index=[0])

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)

        df1=pd.DataFrame(data={'x1':x,
                          'Ca':c1,
                          'Cb':c2,
                          'CaCb':y_iteration,
                          'uncertainty':uncertainty,
                          'gradient':data})

        df.to_csv('inversion_parameters_'+str(index)+'.csv')
        df1.to_csv('inversion_components_'+str(index)+'.csv')

    if constrain=='Y':
        df=pd.DataFrame(data={'y0':m_zero[0],
                          'Bc1':m_zero[1],
                          'W1':m_zero[2],
                          'A1':m_zero[3],
                          'Bc2':m_zero[4],
                          'W2':m_zero[5],
                          'A2':m_zero[6],
                          'var_yzero':COV_P[0,0],
                          'var_W1':COV_P[1,1],
                          'var_A1':COV_P[2,2],
                          'var_W2':COV_P[3,3],
                          'var_A2':COV_P[4,4]},index=[0])

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)

        df1=pd.DataFrame(data={'x1':x,
                          'Ca':c1,
                          'Cb':c2,
                          'CaCb':y_iteration,
                          'uncertainty':uncertainty,
                          'gradient':data})

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)
        df1.to_csv('inversion_components_'+str(index)+'.csv', index=False)


    figure,ax3=plt.subplots(figsize=(5,5))
    ax3.plot(x,y_iteration,color='r',zorder=20)
    ax3.plot(x,data,color='gray',marker='.',label='Data',alpha=0.3)
    ax3.fill_between(x,y_iteration+uncertainty,y_iteration-uncertainty,alpha=0.4,color='royalblue',label='Uncertainty',zorder=20)
    ax3.set_xlabel(r'$B \ (T)$')
    ax3.set_ylabel(r'$\partial{M} / \partial{B} $')
    ax3.grid(alpha=0.5)
    ax3.legend(shadow=True,loc='upper right')
    ax3.set_ylim(min(data)-(min(data)*0.1),(max(data)*0.1)+max(data))
    figure.tight_layout()

    plt.savefig('Final_inversion_model'+str(index)+'.pdf',dpi=300,facecolor='w')

    deltad_f=np.linalg.norm(delta_d,2)

    return m_zero,deltad_f,uncertainty,y_iteration


def Levenberg_Marquardt3(function,X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3,A3,x,data,function2,constrain,condition=1e-5,maxiter=100,index=' sample',sx=2.0E-5,sy=10**-12):
    '''
    Optimization of initial guesses (X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3 and A3),through a Levenberg Marquardt method.As this function plots every single iteration, it
    should be used only to the final model. 

    Parameters: 
        function : phenomenological model based on the Lorentzian function;
        X0 : initial guess (float) for the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1\Bc2\Bc3 : initial guess (float) for the coercive force, where ferromagnetic contribution is zero;
        W1\W2\W3: initial guess (float) for the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1\A2\A3: initial guess (float) for to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        data : a 1D-array containing the values of the dependent variable (lower branch of a magnetic hysteresis)
        constrain: a string, "Y/N", which will determinate if Bc is fixed or optimized.
        condition: a float, being a small value used both in the calculation of the numerical derivatives as a way of residum comparison.
        maxiter: an integer, representing the maximum number of iterations calculated through an inversion process (if the condition is not reach).
        index: a string, the name of the sample. Defaut(' sample').
        sx: a float, the instrumental uncertainty of the field measurements. Defaut (2.0E-5T).
        sy: a float, the instrumental uncertainty of the magnetic moment measurements. Defaut (10**-12 Am²).


    returns: 
        m_zero: a 1D-array with the optimized parameters(X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3 and A3);
        deltad_f:a float, the euclidean norm of the error;
        uncertainty: a 1D-array with the calculated uncertainty at each (x,y) point;
        y_iteration: a 1D-array, a direct model calculated through the optimized parameters (the inverted model).
    


    '''
    a=0.2
    b=1
    damping=1
    itr=0
    bubble_deltad=[]
    bubble_damping=[]
    bubble_damping=[]
    m_zero=np.array([X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3,A3])
    m_cor=np.zeros(np.shape(m_zero))

    if constrain=='N':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_Bc1=(function(m_zero[0],m_zero[1]+condition,m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1]-condition,m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_Bc2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4]+condition,m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4]-condition,m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_Bc3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7]+condition,m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7]-condition,m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]+condition,m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]-condition,m_zero[9],x))/(2*condition)

                    grad_A3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_Bc1,grad_W1,grad_A1,grad_Bc2,grad_W2,grad_A2,grad_Bc3,grad_W3,grad_A3])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1]+par_correct[1],
                                m_zero[2]+par_correct[2],m_zero[3]+par_correct[3],
                                m_zero[4]+par_correct[4],m_zero[5]+par_correct[5],
                                m_zero[6]+par_correct[6],m_zero[7]+par_correct[7],
                                m_zero[8]+par_correct[8],m_zero[9]+par_correct[9]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    c3=function2(0,m_zero[7],m_zero[8],m_zero[9],x)
                    itr+=1
                    figure,ax=plt.subplots(figsize=(5,5))
                    ax.scatter(x,data,alpha=0.2,color='gray',label=r'$Data$')
                    ax.plot(x,y_iteration,color='k',label=r'$C_{a}+C_{b}+C_{c}$')
                    ax.plot(x,c1,label=r'$C_{a}$',color='royalblue')
                    ax.plot(x,c2,label=r'$C_{b}$',color='forestgreen')
                    ax.plot(x,c3,label=r'$C_{c}$',color='m')
                    ax.legend(loc='best',shadow=True)
                    ax.grid(alpha=0.5)
                    ax.set_xlabel(r'$B \ (T)$')
                    ax.set_ylabel(r'$\partial{M} / \partial{B} $')
                    ax.set_title(f'$Loop: \ {itr +1} $')
                    figure.tight_layout()
                    plt.savefig('Inversion_routine '+str(index)+'.pdf',dpi=300,facecolor='w')


                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   

    if constrain=='Y':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]+condition,m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]-condition,m_zero[9],x))/(2*condition)

                    grad_A3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_W1,grad_A1,grad_W2,grad_A2,grad_W3,grad_A3])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1],
                                m_zero[2]+par_correct[1],m_zero[3]+par_correct[2],
                                m_zero[4],m_zero[5]+par_correct[3],
                                m_zero[6]+par_correct[4],m_zero[7],
                                m_zero[8]+par_correct[5],m_zero[9]+par_correct[6]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    c3=function2(0,m_zero[7],m_zero[8],m_zero[9],x)
                    itr+=1
                    figure,ax=plt.subplots(figsize=(5,5))
                    ax.scatter(x,data,alpha=0.2,color='gray',label=r'$Data$')
                    ax.plot(x,y_iteration,color='k',label=r'$C_{a}+C_{b}+C_{c}$')
                    ax.plot(x,c1,label=r'$C_{a}$',color='royalblue')
                    ax.plot(x,c2,label=r'$C_{b}$',color='forestgreen')
                    ax.plot(x,c3,label=r'$C_{c}$',color='m')
                    ax.legend(loc='best',shadow=True)
                    ax.grid(alpha=0.5)
                    ax.set_xlabel(r'$B \ (T)$')
                    ax.set_ylabel(r'$\partial{M} / \partial{B} $')
                    ax.set_title(f'$Loop: \ {itr +1} $')
                    figure.tight_layout()
                    plt.savefig('Inversion_routine '+str(index)+'.pdf',dpi=300,facecolor='w')


                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   

    XY=np.stack((x,data), axis=0)
    COV_XY=np.cov(XY)
    sigma_squared_y=(sy)**2
    sigma_squared_x=(sx)**2
    COV_P=np.linalg.inv((J.T@J))*sigma_squared_y
    gradient_x=np.gradient(x)
    gradient_x[-1]=gradient_x[-2]
    gradient_x[0]=gradient_x[1]
    uncertainty=[]

    if constrain=='N':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+
                                    ((J[i,5]**2)*COV_P[5,5])+((J[i,6]**2)*COV_P[6,6])+
                                    ((J[i,7]**2)*COV_P[7,7])+((J[i,8]**2)*COV_P[8,8])+
                                    ((J[i,9]**2)*COV_P[9,9])+((gradient_x[i]**2)*sigma_squared_x)))
        
        
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])
        m_zero[7]=abs(m_zero[7])
        m_zero[8]=abs(m_zero[8])
        m_zero[9]=abs(m_zero[9])

    if constrain=='Y':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+((J[i,2]**2)*COV_P[2,2])+
                                    ((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+((J[i,5]**2)*COV_P[5,5])+
                                    ((J[i,6]**2)*COV_P[6,6])+((gradient_x[i]**2)*sigma_squared_x)))
              
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])
        m_zero[7]=abs(m_zero[7])
        m_zero[8]=abs(m_zero[8])
        m_zero[9]=abs(m_zero[9])

    figure,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3))
    ax1.plot(bubble_deltad,marker='.',color='gray')
    ax1.set_ylabel('||e||²')
    ax1.set_xlabel('Step')
    ax1.grid(alpha=0.5)
    ax2.plot(bubble_damping,marker='.',color='m')
    ax2.set_ylabel('λ')
    ax2.set_xlabel('Step')
    ax2.grid(alpha=0.5)
    figure.tight_layout()


    plt.savefig('Optmization_parameters '+str(index)+'.pdf',dpi=300,facecolor='w')
    if constrain=='N':
        df=pd.DataFrame(data={'y0':m_zero[0],
                          'Bc1':m_zero[1],
                          'W1':m_zero[2],
                          'A1':m_zero[3],
                          'Bc2':m_zero[4],
                          'W2':m_zero[5],
                          'A2':m_zero[6],
                          'Bc3':m_zero[7],
                          'W3':m_zero[8],
                          'A3':m_zero[9],
                          'var_yzero':COV_P[0,0],
                          'var_Bc1':COV_P[1,1],
                          'var_W1':COV_P[2,2],
                          'var_A1':COV_P[3,3],
                          'var_Bc2':COV_P[4,4],
                          'var_W2':COV_P[5,5],
                          'var_A2':COV_P[6,6],
                          'var_Bc3':COV_P[7,7],
                          'var_W3':COV_P[8,8],
                          'var_A3':COV_P[9,9]},index=[0])

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)

        df1=pd.DataFrame(data={'x1':x,
                          'Ca':c1,
                          'Cb':c2,
                          'Cc':c3,
                          'CaCbCc':y_iteration,
                          'uncertainty':uncertainty,
                          'gradient':data})

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)
        df1.to_csv('inversion_components_'+str(index)+'.csv', index=False)

    if constrain=='Y':
        df=pd.DataFrame(data={'y0':m_zero[0],
                          'Bc1':m_zero[1],
                          'W1':m_zero[2],
                          'A1':m_zero[3],
                          'Bc2':m_zero[4],
                          'W2':m_zero[5],
                          'A2':m_zero[6],
                          'Bc3':m_zero[7],
                          'W3':m_zero[8],
                          'A3':m_zero[9],
                          'var_yzero':COV_P[0,0],
                          'var_W1':COV_P[1,1],
                          'var_A1':COV_P[2,2],
                          'var_W2':COV_P[3,3],
                          'var_A2':COV_P[4,4],
                          'var_W3':COV_P[5,5],
                          'var_A3':COV_P[6,6]},index=[0])

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)

        df1=pd.DataFrame(data={'x1':x,
                          'Ca':c1,
                          'Cb':c2,
                          'Cc':c3,
                          'CaCbCc':y_iteration,
                          'uncertainty':uncertainty,
                          'gradient':data})

        df.to_csv('inversion_parameters_'+str(index)+'.csv', index=False)
        df1.to_csv('inversion_components_'+str(index)+'.csv', index=False)

    figure,ax3=plt.subplots(figsize=(5,5))
    ax3.plot(x,y_iteration,color='r',zorder=20)
    ax3.plot(x,data,color='gray',marker='.',label='Data',alpha=0.3)
    ax3.fill_between(x,y_iteration+uncertainty,y_iteration-uncertainty,alpha=0.4,color='royalblue',label='Uncertainty',zorder=20)
    ax3.set_xlabel(r'$B \ (T)$')
    ax3.set_ylabel(r'$\partial{M} / \partial{B} $')
    ax3.grid(alpha=0.5)
    ax3.legend(shadow=True,loc='upper right')
    ax3.set_ylim(min(data)-(min(data)*0.1),(max(data)*0.1)+max(data))
    figure.tight_layout()

    plt.savefig('Final_inversion_model'+str(index)+'.pdf',dpi=300,facecolor='w')

    deltad_f=np.linalg.norm(delta_d,2)

    return m_zero,deltad_f,uncertainty,y_iteration

 



def LV_whithout_plot3(function,X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3,A3,x,data,function2,constrain,condition=1e-5,maxiter=100,index=' sample',sx=2.0E-5,sy=10**-12):
    '''
    Optimization of initial guesses (X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3 and A3),through a Levenberg Marquardt method.This function does not show plots through the iterations.

    Parameters: 
        function : phenomenological model based on the Lorentzian function;
        X0 : initial guess (float) for the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1\Bc2\Bc3 : initial guess (float) for the coercive force, where ferromagnetic contribution is zero;
        W1\W2\W3: initial guess (float) for the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1\A2\A3: initial guess (float) for to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        data : a 1D-array containing the values of the dependent variable (lower branch of a magnetic hysteresis)
        constrain: a string, "Y/N", which will determinate if Bc is fixed or optimized.
        condition: a float, being a small value used both in the calculation of the numerical derivatives as a way of residum comparison.
        maxiter: an integer, representing the maximum number of iterations calculated through an inversion process (if the condition is not reach).
        index: a string, the name of the sample. Defaut(' sample').
        sx: a float, the instrumental uncertainty of the field measurements. Defaut (2.0E-5T).
        sy: a float, the instrumental uncertainty of the magnetic moment measurements. Defaut (10**-12 Am²).


    returns: 
        m_zero: a 1D-array with the optimized parameters(X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3 and A3);
        deltad_f:a float, the euclidean norm of the error;
        y_iteration: a 1D-array, a direct model calculated through the optimized parameters (the inverted model).
    


    '''
    a=0.2
    b=1
    damping=1
    itr=0
    bubble_deltad=[]
    bubble_damping=[]
    bubble_damping=[]
    m_zero=np.array([X0,Bc1,W1,A1,Bc2,W2,A2,Bc3,W3,A3])
    m_cor=np.zeros(np.shape(m_zero))
    if constrain=='N':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_Bc1=(function(m_zero[0],m_zero[1]+condition,m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1]-condition,m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_Bc2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4]+condition,m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4]-condition,m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_Bc3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7]+condition,m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7]-condition,m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]+condition,m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]-condition,m_zero[9],x))/(2*condition)

                    grad_A3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_Bc1,grad_W1,grad_A1,grad_Bc2,grad_W2,grad_A2,grad_Bc3,grad_W3,grad_A3])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1]+par_correct[1],
                                m_zero[2]+par_correct[2],m_zero[3]+par_correct[3],
                                m_zero[4]+par_correct[4],m_zero[5]+par_correct[5],
                                m_zero[6]+par_correct[6],m_zero[7]+par_correct[7],
                                m_zero[8]+par_correct[8],m_zero[9]+par_correct[9]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    c3=function2(0,m_zero[7],m_zero[8],m_zero[9],x)
                    itr+=1
                    
                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   

    if constrain=='Y':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,m_zero[7],m_zero[8],m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,m_zero[7],m_zero[8],m_zero[9],x))/(2*condition)

                    grad_W3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]+condition,m_zero[9],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8]-condition,m_zero[9],x))/(2*condition)

                    grad_A3=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_W1,grad_A1,grad_W2,grad_A2,grad_W3,grad_A3])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1],
                                m_zero[2]+par_correct[1],m_zero[3]+par_correct[2],
                                m_zero[4],m_zero[5]+par_correct[3],
                                m_zero[6]+par_correct[4],m_zero[7],
                                m_zero[8]+par_correct[5],m_zero[9]+par_correct[6]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9]])
                    m_zero=m_cor
                       
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],m_zero[7],m_zero[8],m_zero[9],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    c3=function2(0,m_zero[7],m_zero[8],m_zero[9],x)
                    itr+=1
                    
                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   

    XY=np.stack((x,data), axis=0)
    COV_XY=np.cov(XY)
    sigma_squared_y=(sy)**2
    sigma_squared_x=(sx)**2
    COV_P=np.linalg.inv((J.T@J))*sigma_squared_y
    gradient_x=np.gradient(x)
    gradient_x[-1]=gradient_x[-2]
    gradient_x[0]=gradient_x[1]
    uncertainty=[]

    if constrain=='N':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+
                                    ((J[i,5]**2)*COV_P[5,5])+((J[i,6]**2)*COV_P[6,6])+
                                    ((J[i,7]**2)*COV_P[7,7])+((J[i,8]**2)*COV_P[8,8])+
                                    ((J[i,9]**2)*COV_P[9,9])+((gradient_x[i]**2)*sigma_squared_x)))
        
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])
        m_zero[7]=abs(m_zero[7])
        m_zero[8]=abs(m_zero[8])
        m_zero[9]=abs(m_zero[9])

    if constrain=='Y':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+((J[i,2]**2)*COV_P[2,2])+
                                    ((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+((J[i,5]**2)*COV_P[5,5])+
                                    ((J[i,6]**2)*COV_P[6,6])+((gradient_x[i]**2)*sigma_squared_x)))
              
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])
        m_zero[7]=abs(m_zero[7])
        m_zero[8]=abs(m_zero[8])
        m_zero[9]=abs(m_zero[9])

    deltad_f=np.linalg.norm(delta_d,2)
    return deltad_f,y_iteration,m_zero


def LV_whithout_plot2(function,X0,Bc1,W1,A1,Bc2,W2,A2,x,data,function2,constrain,condition=1e-5,maxiter=100,index=' sample',sx=2.0E-5,sy=10**-12):
    '''
    Optimization of initial guesses (X0,Bc1,W1,A1,Bc2,W2 and A2),through a Levenberg Marquardt method.As this function plots every single iteration, it
    should be used only to the final model. 

    Parameters: 
        function : phenomenological model based on the Lorentzian function;
        X0 : initial guess (float) for the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1\Bc2 : initial guess (float) for the coercive force, where ferromagnetic contribution is zero;
        W1\W2: initial guess (float) for the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1\A2: initial guess (float) for to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        data : a 1D-array containing the values of the dependent variable (lower branch of a magnetic hysteresis)
        constrain: a string, "Y/N", which will determinate if Bc is fixed or optimized.
        condition: a float, being a small value used both in the calculation of the numerical derivatives as a way of residum comparison.
        maxiter: an integer, representing the maximum number of iterations calculated through an inversion process (if the condition is not reach).
        index: a string, the name of the sample. Defaut(' sample').
        sx: a float, the instrumental uncertainty of the field measurements. Defaut (2.0E-5T).
        sy: a float, the instrumental uncertainty of the magnetic moment measurements. Defaut (10**-12 Am²).


    returns: 
        m_zero: a 1D-array with the optimized parameters(X0,Bc1,W1,A1,Bc2,W2 and A2);
        deltad_f:a float, the euclidean norm of the error;
        y_iteration: a 1D-array, a direct model calculated through the optimized parameters (the inverted model).
    


    '''
    a=0.2
    b=1
    damping=1
    itr=0
    bubble_deltad=[]
    bubble_damping=[]
    bubble_damping=[]
    m_zero=np.array([X0,Bc1,W1,A1,Bc2,W2,A2])
    m_cor=np.zeros(np.shape(m_zero))
    if constrain=='N':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_Bc1=(function(m_zero[0],m_zero[1]+condition,m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1]-condition,m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_Bc2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4]+condition,m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4]-condition,m_zero[5],m_zero[6],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_Bc1,grad_W1,grad_A1,grad_Bc2,grad_W2,grad_A2])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1]+par_correct[1],
                                m_zero[2]+par_correct[2],m_zero[3]+par_correct[3],
                                m_zero[4]+par_correct[4],m_zero[5]+par_correct[5],
                                m_zero[6]+par_correct[6]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    
                    itr+=1
                    
                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   
    if constrain=='Y':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],m_zero[4],
                    m_zero[5],m_zero[6],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,
                                      m_zero[4],m_zero[5],m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,
                                                                                m_zero[4],m_zero[5],m_zero[6],x))/(2*condition)

                    grad_W2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5]+condition,m_zero[6],x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5]-condition,m_zero[6],x))/(2*condition)

                    grad_A2=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                      m_zero[4],m_zero[5],m_zero[6]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                                                                                m_zero[4],m_zero[5],m_zero[6]-condition,x))/(2*condition)

                    J=np.column_stack([grad_yzero,grad_W1,grad_A1,grad_W2,grad_A2])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1],
                                m_zero[2]+par_correct[1],m_zero[3]+par_correct[2],
                                m_zero[4],m_zero[5]+par_correct[3],
                                m_zero[6]+par_correct[4]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],
                        m_zero[4],m_zero[5],m_zero[6],x)
                    c1=function2(0,m_zero[1],m_zero[2],m_zero[3],x)
                    c2=function2(0,m_zero[4],m_zero[5],m_zero[6],x)
                    
                    itr+=1
                    
                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break  
    XY=np.stack((x,data), axis=0)
    COV_XY=np.cov(XY)
    sigma_squared_y=(sy)**2
    sigma_squared_x=(sx)**2
    COV_P=np.linalg.inv((J.T@J))*sigma_squared_y
    gradient_x=np.gradient(x)
    gradient_x[-1]=gradient_x[-2]
    gradient_x[0]=gradient_x[1]
    uncertainty=[]

    if constrain=='N':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+
                                    ((J[i,5]**2)*COV_P[5,5])+((J[i,6]**2)*COV_P[6,6])+((gradient_x[i]**2)*sigma_squared_x)))
        
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])

    if constrain=='Y':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+
            ((J[i,1]**2)*COV_P[1,1])+((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((J[i,4]**2)*COV_P[4,4])+((gradient_x[i]**2)*sigma_squared_x)))
        

        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])
        m_zero[4]=abs(m_zero[4])
        m_zero[5]=abs(m_zero[5])
        m_zero[6]=abs(m_zero[6])
    
    deltad_f=np.linalg.norm(delta_d,2)
    return deltad_f,y_iteration,m_zero




def LV_whithout_plot1(function,X0,Bc1,W1,A1,x,data,constrain,condition=1e-5,maxiter=100,index=' sample',sx=2.0E-5,sy=10**-12):
    '''
    Optimization of initial guesses (X0,Bc1,W1 and A1),through a Levenberg Marquardt method. This function does not show plots through the iterations.

    Parameters: 
        function : phenomenological model based on the Lorentzian function;
        X0 : initial guess (float) for the high-field magnetic susceptibility (diamagnetic or/and paramagnetic);
        Bc1 : initial guess (float) for the coercive force, where ferromagnetic contribution is zero;
        W1: initial guess (float) for the Lorentzian Full-Width-Half-Maximum (FWHM), or 2(σ);
        A1: initial guess (float) for to integral under the Lorentzian component;
        x : a 1D-array containing the values of the applied field
        data : a 1D-array containing the values of the dependent variable (lower branch of a magnetic hysteresis)
        constrain: a string, "Y/N", which will determinate if Bc is fixed or optimized.
        condition: a float, being a small value used both in the calculation of the numerical derivatives as a way of residum comparison.
        maxiter: an integer, representing the maximum number of iterations calculated through an inversion process (if the condition is not reach).
        index: a string, the name of the sample. Defaut(' sample').
        sx: a float, the instrumental uncertainty of the field measurements. Defaut (2.0E-5T).
        sy: a float, the instrumental uncertainty of the magnetic moment measurements. Defaut (10**-12 Am²).


    returns: 
        m_zero: a 1D-array with the optimized parameters(X0,Bc1,W1 and A1);
        deltad_f:a float, the euclidean norm of the error;
        y_iteration: a 1D-array, a direct model calculated through the optimized parameters (the inverted model).
    


    '''

    a=0.2
    b=1
    damping=1
    itr=0
    bubble_deltad=[]
    bubble_damping=[]
    bubble_damping=[]
    m_zero=np.array([X0,Bc1,W1,A1])
    m_cor=np.zeros(np.shape(m_zero))

    if constrain=='N':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],x))/(2*condition)
                    grad_Bc1=(function(m_zero[0],m_zero[1]+condition,m_zero[2],m_zero[3],x)-function(m_zero[0],m_zero[1]-condition,m_zero[2],m_zero[3],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,x))/(2*condition)
            
                    J=np.column_stack([grad_yzero,grad_Bc1,grad_W1,grad_A1])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1]+par_correct[1],
                                m_zero[2]+par_correct[2],m_zero[3]+par_correct[3]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                    c1=function(0,m_zero[1],m_zero[2],m_zero[3],x)
                    itr+=1
                    
                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   

    
    if constrain=='Y':
        for i in range(maxiter):
            if i==0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
            elif i>0:
                delta_d=data-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                bubble_deltad=np.append(bubble_deltad,np.linalg.norm(delta_d,2))
                bubble_damping=np.append(bubble_damping,damping)
                if (np.linalg.norm(delta_d))<=bubble_deltad[i-1]:
                    grad_yzero=(function(m_zero[0]+condition,m_zero[1],m_zero[2],m_zero[3],x)-function(m_zero[0]-condition,m_zero[1],m_zero[2],m_zero[3],x))/(2*condition)
                    grad_W1=(function(m_zero[0],m_zero[1],m_zero[2]+condition,m_zero[3],x)-function(m_zero[0],m_zero[1],m_zero[2]-condition,m_zero[3],x))/(2*condition)
                    grad_A1=(function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]+condition,x)-function(m_zero[0],m_zero[1],m_zero[2],m_zero[3]-condition,x))/(2*condition)
            
                    J=np.column_stack([grad_yzero,grad_W1,grad_A1])
                    damping=(1/a)*damping
                    par_correct=(np.linalg.inv((J.T@J)+(damping*np.identity(np.shape(J.T@J)[0]))))@J.T@delta_d
                    m_cor=np.array([m_zero[0]+par_correct[0],m_zero[1],
                                m_zero[2]+par_correct[1],m_zero[3]+par_correct[2]])
                    m_zero=m_cor
                    

                elif (np.linalg.norm(delta_d))>bubble_deltad[i-1]:
                    damping=(b*damping)
                    m_cor=np.array([m_zero[0],m_zero[1],m_zero[2],m_zero[3]])
                    m_zero=m_cor
                    
                if i>1:
                    y_iteration=function(m_zero[0],m_zero[1],m_zero[2],m_zero[3],x)
                    c1=function(0,m_zero[1],m_zero[2],m_zero[3],x)
                    itr+=1
                    
                    tolerance=(np.linalg.norm(par_correct,2)/np.linalg.norm(delta_d,2))
                    
                    if tolerance<=condition*2 or itr==maxiter: #testing the condition
                        break   


    XY=np.stack((x,data), axis=0)
    COV_XY=np.cov(XY)
    sigma_squared_y=(sy)**2
    sigma_squared_x=(sx)**2
    COV_P=np.linalg.inv((J.T@J))*sigma_squared_y
    gradient_x=np.gradient(x)
    gradient_x[-1]=gradient_x[-2]
    gradient_x[0]=gradient_x[1]
    uncertainty=[]

    if constrain=='N':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((J[i,3]**2)*COV_P[3,3])+((gradient_x[i]**2)*sigma_squared_x)))
            
        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])

    if constrain=='Y':
        for i in range(np.size(x)):
            uncertainty=np.append(uncertainty,np.sqrt(((J[i,0]**2)*COV_P[0,0])+((J[i,1]**2)*COV_P[1,1])+
            ((J[i,2]**2)*COV_P[2,2])+((gradient_x[i]**2)*sigma_squared_x)))

        m_zero[1]=abs(m_zero[1])
        m_zero[2]=abs(m_zero[2])
        m_zero[3]=abs(m_zero[3])

    deltad_f=np.linalg.norm(delta_d,2)

    return deltad_f,y_iteration,m_zero

