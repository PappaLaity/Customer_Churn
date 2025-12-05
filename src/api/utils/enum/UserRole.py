#src/api/utils/enum/UserRole.py
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    SUPERVISOR = "supervisor"
    GUEST = "guest"
