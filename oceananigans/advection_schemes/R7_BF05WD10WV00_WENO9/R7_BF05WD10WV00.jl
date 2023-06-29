using Random
using Printf
using Plots
using JLD2

using Oceananigans
using Oceananigans.Units: minute, minutes, hour, hours

# ## The grid
#
grid = RectilinearGrid(GPU();
                       size = (256, 256, 256),
                     extent = (320, 320, 163.84))

# ## Buoyancy that depends on temperature and salinity
#
# We use the `SeawaterBuoyancy` model with a linear equation of state,

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=8e-4))

# where ``\alpha`` and ``\beta`` are the thermal expansion and haline contraction
# coefficients for temperature and salinity.
#
# ## Boundary conditions
#

Qʰ = 5  # W m⁻², surface heat flux
ρₒ = 1000 # kg m⁻³, water density
cᴾ = 4200 # J K⁻¹ kg⁻¹, typical heat capacity for seawater

Qᵀ = Qʰ / (ρₒ * cᴾ) # K m s⁻¹, surface _temperature_ flux

# Finally, we impose a temperature gradient `dTdz` both initially and at the
# bottom of the domain, culminating in the boundary conditions on temperature,

dTdz = 0.01 # K m⁻¹

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# Note that a positive temperature flux at the surface of the ocean
# implies cooling. This is because a positive temperature flux implies
# that temperature is fluxed upwards, out of the ocean.
#
# For the velocity field, we imagine a wind blowing over the ocean surface
# with an average velocity at 10 meters `u₁₀`, and use a drag coefficient `cᴰ`
# to estimate the kinematic stress (that is, stress divided by density) exerted
# by the wind on the ocean:

# u₁₀ = 10.0    # m s⁻¹, average wind velocity 10 meters above the ocean
# cᴰ = 2.5e-3 # dimensionless drag coefficient
# ρₐ = 1.225  # kg m⁻³, average density of air at sea-level
# Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

ustar = 0.012 # m s⁻¹, friction velocity
Qᵘ = - ustar*ustar # m² s⁻²

# The boundary conditions on `u` are thus

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

# For salinity, `S`, we impose an evaporative flux of the form

# @inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S # [salinity unit] m s⁻¹
# nothing # hide

# where `S` is salinity. We use an evporation rate of 1 millimeter per hour,
# evaporation_rate = 1e-3 / hour # m s⁻¹
# evaporation_rate = 0.0 # m s⁻¹

# We build the `Flux` evaporation `BoundaryCondition` with the function `Qˢ`,
# indicating that `Qˢ` depends on salinity `S` and passing
# the parameter `evaporation_rate`,

# evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=evaporation_rate)

# The full salinity boundary conditions are

# S_bcs = FieldBoundaryConditions(top=evaporation_bc)
Qˢ = 0.0
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ))

# initial condition
S₀ = 35.0
hb₀ = 42 # m
T₀ = 11.85
T0(z) = z < - hb₀ ? T₀ + dTdz * (z + hb₀) : T₀

#####
##### Sponge layer
#####

gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=1/hour, mask=gaussian_mask)

T_sponge = Relaxation(rate = 1/hour,
                      target = LinearTarget{:z}(intercept=T₀+dTdz*hb₀, gradient=dTdz),
                      mask = gaussian_mask)
S_sponge = Relaxation(rate = 1/hour,
                      target = S₀,
                      mask = gaussian_mask)

# ## Model instantiation
#
# We fill in the final details of the model here: upwind-biased 5th-order
# advection for momentum and tracers, 3rd-order Runge-Kutta time-stepping,
# Coriolis forces, and the `AnisotropicMinimumDissipation` closure
# for large eddy simulation to model the effect of turbulent motions at
# scales smaller than the grid scale that we cannot explicitly resolve.

model = NonhydrostaticModel(
                advection = WENO(order=9),
              timestepper = :RungeKutta3,
                     grid = grid,
                  tracers = (:T, :S),
                 coriolis = FPlane(f=1.028e-4),
                 buoyancy = buoyancy,
                  closure = AnisotropicMinimumDissipation(),
      boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs),
                  forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge, S=S_sponge))

# Notes:
#
# * To use the Smagorinsky-Lilly turbulence closure (with a constant model coefficient) rather than
#   `AnisotropicMinimumDissipation`, use `closure = SmagorinskyLilly()` in the model constructor.
#
# * To change the `architecture` to `GPU`, replace `architecture = CPU()` with
#   `architecture = GPU()`.

# ## Initial conditions
#
# Our initial condition for temperature consists of a linear stratification superposed with
# random noise damped at the walls, while our initial condition for velocity consists
# only of random noise.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = T0(z) + dTdz * model.grid.Lz * 1e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)

## `set!` the `model` fields using functions or constants:
set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=S₀)

# ## Setting up a simulation
#
# We set-up a simulation with an initial time-step of 10 seconds
# that stops at 40 minutes, with adaptive time-stepping and progress printing.

simulation = Simulation(model, Δt=10.0, stop_time=28hours)

# The `TimeStepWizard` helps ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 1.0.

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# Nice progress messaging is helpful:

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim),
                                prettytime(sim),
                                prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

# We then set up the simulation:

# ## Output
#
# We use the `JLD2OutputWriter` to save ``x, z`` slices of the velocity fields,
# tracer fields, and eddy diffusivities. The `prefix` keyword argument
# to `JLD2OutputWriter` indicates that output will be saved in
# `ocean_wind_mixing_and_convection.jld2`.

## Create a NamedTuple with eddy viscosity
eddy_viscosity = (; νₑ = model.diffusivity_fields.νₑ)

simulation.output_writers[:slices] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices.jld2",
                          indices = (:, grid.Ny/2, :),
                         schedule = TimeInterval(10minute),
               overwrite_existing = true)

fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivity_fields.νₑ,))
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                     filename = "fields.jld2",
                     schedule = TimeInterval(1hours),
           overwrite_existing = true)

u, v, w = model.velocities
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T = Average(model.tracers.T, dims=(1, 2))
wt = Average(w * model.tracers.T, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))
ww = Average(w * w, dims=(1, 2))
uu = Average((u-Field(U)) * (u-Field(U)), dims=(1, 2))
vv = Average((v-Field(V)) * (v-Field(V)), dims=(1, 2))
w3 = Average(w^3, dims=(1, 2))
tt = Average((model.tracers.T-Field(T)) * (model.tracers.T-Field(T)), dims=(1, 2))

simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (u=U, v=V, T=T, wt=wt, wu=wu, wv=wv, ww=ww, vv=vv, uu=uu, w3=w3, tt=tt),
    # JLD2OutputWriter(model, (u=U, v=V, T=T, wt=wt, wu=wu, wv=wv, ww=ww),
                     schedule = TimeInterval(3minutes),
                     filename = "averages.jld2",
           overwrite_existing = true)

# We're ready:

run!(simulation)

