# LZ-unmix
LZ-unmix is an open-sourced package, in python language (ipyfile), to separate susceptibility components of distorted hysteresis curves through a phenomenological model (Laurentzian fits), developed based on the work of Vasquez & Fazzito (2020). 
The LZ-unmix package allows the user to adjust a direct model of up to three ferromagnetic components and a dia/paramagnetic contribution. Optimization of all of the parameters is achieved through least squares fit (Levemberg-Marquadt method) providing an uncertainty envelope for the inversion. For each ferromagnetic component, it is possible to calculate magnetization saturation (Ms), magnetization saturation of remanence (Mrs) and the mean coercivity (Bc), as well as their dispersion (W) and their contribution to the spectrum (A)
