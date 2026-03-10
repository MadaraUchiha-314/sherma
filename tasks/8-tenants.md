- A2A has the contruct of a tenant
- Add the construct of a tenant id which is a string
- All registeries should be initialized to a scope of a tenant id
    - All registeries must be singleton to that scope
- Add tenant id to entities/base.py

- Any search of an entity in a registry should have a tenant id associated with it for isolation
- Entities registered for a tenant shouldn't be accessible to another tenant unless that tenant also has that entity registered.