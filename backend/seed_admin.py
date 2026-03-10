import asyncio
from database.db import init_db, AsyncSessionLocal
from database.models import User
from utils.security import hash_password

async def seed():
    await init_db()
    async with AsyncSessionLocal() as db:
        admin = User(
            name="Admin",
            email="admin@example.com",
            password_hash=hash_password("admin123"),
            role="admin",
        )
        db.add(admin)
        await db.commit()
        print("Admin created: admin@example.com / admin123")

asyncio.run(seed())