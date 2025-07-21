from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
from sqlalchemy import func
import bcrypt

Base = declarative_base()


class User(Base):
    __tablename__ = "Users"
    id = Column(String(255), primary_key=True)
    password = Column(LargeBinary, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    registration_date = Column(DateTime,nullable=False)

    # Relationship to History
    histories = relationship("History", back_populates="user")

    def __init__(
        self,
        username,
        password,
        first_name,
        last_name,
        date_of_birth,
        registration_date,
    ):
        self.id = username
        # Store password securely
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.first_name = first_name
        self.last_name = last_name
        self.date_of_birth = date_of_birth
        self.registration_date = registration_date

    def add_history(self, media_item_id):
        new_history = History(
            user_id=self.id,
            media_item_id=media_item_id,
            viewtime=datetime.datetime.now()
        )
        self.histories.append(new_history)

    def sum_title_length (self):
        return sum(history.mediaitem.title_length for history in self.histories if history.mediaitem)


class MediaItem(Base):
    __tablename__ = "MediaItems"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    prod_year = Column(Integer, nullable=False)
    title_length = Column(Integer, nullable=False)

    def __init__(self, title, prod_year, title_length):
        self.title = title
        self.prod_year = prod_year
        self.title_length = len(title) if title_length is None else title_length


class History(Base):
    __tablename__ = "History"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("Users.id"), nullable=False)
    media_item_id = Column(Integer, ForeignKey("MediaItems.id"), nullable=False)
    viewtime = Column(DateTime, nullable=False)

    # Relationship to User
    user = relationship("User", back_populates="histories")
    # Relationship to MediaItem
    mediaitem = relationship("MediaItem")

    def __init__(self, user_id, media_item_id, viewtime):
        self.user_id = user_id
        self.media_item_id = media_item_id
        self.viewtime = viewtime


class Repository:
    def __init__(self, model_class):
        self.model_class=model_class

    def get_by_id(self, session, entity_id):
        return session.query(self.model_class).filter(self.model_class.id == entity_id).first()
    
    def get_all(self,session):
        return session.query(self.model_class).all()
    
    def delete(self,session, entity):
        session.delete(entity)

    def add(self, session, entity):
        session.add(entity)

class UserRepository(Repository):
    def __init__(self):
        super().__init__(User)

    def validateUser(self, session, username: str, password: str) -> bool:
        # Fetch the user by username
        user = session.query(User).filter(User.id == username).first()
        if user:
            # Check if the provided password matches the stored encrypted password
            return bcrypt.checkpw(password.encode('utf-8'), user.password)
        return False

    def getNumberOfRegistredUsers(self, session, n: int) -> int:
        # Calculate the date n days ago
        n_days_ago = datetime.datetime.now() - datetime.timedelta(days=n)
        # Query the database to count users registered on or after n_days_ago
        return session.query(User).filter(User.registration_date >= n_days_ago).count()

class ItemRepository(Repository):
    def __init__(self):
        super().__init__(MediaItem)

    def getTopNItems(self, session, top_n: int) -> list:
        # Query the MediaItems table, order by id ascending, and limit to top_n results
        return session.query(MediaItem).order_by(MediaItem.id.asc()).limit(top_n).all()
    

    
class UserService:
    def __init__(self, session, user_repo: UserRepository):
        self.user_repo = user_repo
        self.session = session

    def create_user(self, username, password, first_name, last_name, date_of_birth):
        # Create a new user with the current registration date
        new_user = User(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            registration_date=datetime.datetime.now(),
        )
        # Add the user to the database
        self.user_repo.add(self.session, new_user)
        self.session.commit()

    def add_history_to_user(self, username, media_item_id):
        # Retrieve the user by username
        user = self.session.query(User).filter(User.id == username).first()
        if not user:
            raise ValueError("User not found")
        # Add history to the user
        user.add_history(media_item_id)
        self.session.commit()
    
    def validateUser(self, username: str, password: str) -> bool:
        # Validate user credentials using UserRepository
        return self.user_repo.validateUser(self.session, username,password)

    def getNumberOfRegistredUsers(self, n: int) -> int:
        # Get the number of users registered in the last n days using UserRepository
        return self.user_repo.getNumberOfRegistredUsers(self.session, n)


    def sum_title_length_to_user(self, username):
        # Retrieve the user by username
        user = self.session.query(User).filter(User.id == username).first()
        if not user:
            raise ValueError("User not found")
        # Calculate the total title length for the user's histories
        return user.sum_title_length()


    def get_all_users(self):
        # Retrieve all users using UserRepository
        return self.user_repo.get_all(self.session)
    

class ItemService:
    def __init__(self, session, item_repo:ItemRepository):
        self.item_repo = item_repo
        self.session = session

    def create_item(self, title, prod_year):
        # Create a new MediaItem with the given details
        new_item = MediaItem(
            title=title,
            prod_year=prod_year,
            title_length=len(title)
        )
        # Add the item to the database
        self.item_repo.add(self.session, new_item)
        self.session.commit()

#username = ''
#password = ''
#connection_string = f"mssql+pyodbc://{username}:{password}@132.72.64.124/{username}?driver=ODBC+Driver+17+for+SQL+Server"
#engine = create_engine(connection_string)
#Base.metadata.create_all(engine)
#session = sessionmaker(bind=engine)()
