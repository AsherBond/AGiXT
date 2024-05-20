import os
from fastapi import APIRouter, Request, HTTPException, Header, Depends
from MagicalAuth import MagicalAuth
from ApiClient import verify_api_key, DB_CONNECTED, get_api_client, is_admin
from Models import User, Register, Login, UserInfo

app = APIRouter()
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")

if SENDGRID_API_KEY != "" and DB_CONNECTED:

    @app.post("/register")
    def register(
        register: Register,
    ):
        auth = MagicalAuth(email=register.email)
        mfa_token = auth.register(
            first_name=register.first_name,
            last_name=register.last_name,
            company_name=register.company_name,
            job_title=register.job_title,
        )
        return {"mfa_token": mfa_token}

    @app.post("/login")
    def login(login: Login, request: Request):
        auth = MagicalAuth(email=login.email, token=login.token)
        user = auth.login(ip_address=request.client.host)
        return {
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "company_name": user.company_name,
            "job_title": user.job_title,
        }

    @app.post("/send_magic_link")
    def send_magic_link(login: Login, request: Request):
        auth = MagicalAuth(email=login.email)
        magic_link = auth.send_magic_link(
            otp=login.token, ip_address=request.client.host
        )
        return {"message": magic_link}

    @app.put("/update_user")
    def update_user(update: UserInfo, login: Login):
        auth = MagicalAuth(email=login.email, token=login.token)
        user = auth.login()
        user.first_name = update.first_name
        user.last_name = update.last_name
        user.company_name = update.company_name
        user.job_title = update.job_title
        return {"message": "User updated successfully."}

    # Delete user
    @app.delete("/delete_user")
    def delete_user(login: Login):
        user = MagicalAuth(email=login.email, token=login.token)
        user.delete_user()
        return {"message": "User deleted successfully."}


if DB_CONNECTED:
    from db.User import create_user

    @app.post("/api/user", tags=["User"])
    async def createuser(
        account: User, authorization: str = Header(None), user=Depends(verify_api_key)
    ):
        if is_admin(email=user, api_key=authorization) != True:
            raise HTTPException(status_code=403, detail="Access Denied")
        ApiClient = get_api_client(authorization=authorization)
        return create_user(
            api_key=authorization,
            email=account.email,
            role="user",
            agent_name=account.agent_name,
            settings=account.settings,
            commands=account.commands,
            training_urls=account.training_urls,
            github_repos=account.github_repos,
            ApiClient=ApiClient,
        )

    @app.post("/api/admin", tags=["User"])
    async def createadmin(
        account: User, authorization: str = Header(None), user=Depends(verify_api_key)
    ):
        if is_admin(email=user, api_key=authorization) != True:
            raise HTTPException(status_code=403, detail="Access Denied")
        ApiClient = get_api_client(authorization=authorization)
        return create_user(
            api_key=authorization,
            email=account.email,
            role="admin",
            agent_name=account.agent_name,
            settings=account.settings,
            commands=account.commands,
            training_urls=account.training_urls,
            github_repos=account.github_repos,
            ApiClient=ApiClient,
        )
