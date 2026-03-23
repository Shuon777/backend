from .interfaces import SchemaRepository

class RecreateDatabaseUseCase:
    def __init__(self, schema_repo: SchemaRepository):
        self._schema_repo = schema_repo

    def execute(self) -> None:
        self._schema_repo.drop_all()
        self._schema_repo.create_all()