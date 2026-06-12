"""GraphQL queries for repository acquisition."""

GET_REPOSITORY_QUERY = """
query GetRepository($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    name
    nameWithOwner
    description
    url
    homepageUrl
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    
    watchers {
      totalCount
    }
    issues(states: OPEN) {
      totalCount
    }
    pullRequests(states: OPEN) {
      totalCount
    }
    
    repositoryTopics(first: 20) {
      nodes {
        topic {
          name
        }
      }
    }
    
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      edges {
        size
        node {
          name
        }
      }
    }
    
    licenseInfo {
      name
      spdxId
    }
    
    owner {
      login
      __typename
    }
    
    defaultBranchRef {
      name
      target {
        ... on Commit {
          history(first: 30) {
            nodes {
              committedDate
            }
          }
        }
      }
    }
    
    readme1: object(expression: "HEAD:README.md") {
      ... on Blob {
        text
      }
    }
    readme2: object(expression: "HEAD:readme.md") {
      ... on Blob {
        text
      }
    }
    readme3: object(expression: "HEAD:README.rst") {
      ... on Blob {
        text
      }
    }
    readme4: object(expression: "HEAD:README.txt") {
      ... on Blob {
        text
      }
    }
    readme5: object(expression: "HEAD:README") {
      ... on Blob {
        text
      }
    }

    stargazers(last: 100) {
      edges {
        starredAt
      }
    }
    
    mentionableUsers(first: 100) {
      totalCount
      nodes {
        login
      }
    }
  }
  
  rateLimit {
    cost
    remaining
    resetAt
  }
}
"""

def build_batch_query(repos: list[tuple[str, str]]) -> str:
    """Dynamically builds a GraphQL query for multiple repositories using aliases."""
    query_parts = ["query GetRepositoriesBatch {"]
    for i, (owner, name) in enumerate(repos):
        alias = f"repo_{i}"
        part = f"""
  {alias}: repository(owner: "{owner}", name: "{name}") {{
    id
    name
    nameWithOwner
    description
    url
    homepageUrl
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    
    watchers {{
      totalCount
    }}
    issues(states: OPEN) {{
      totalCount
    }}
    pullRequests(states: OPEN) {{
      totalCount
    }}
    
    repositoryTopics(first: 20) {{
      nodes {{
        topic {{
          name
        }}
      }}
    }}
    
    languages(first: 10, orderBy: {{field: SIZE, direction: DESC}}) {{
      edges {{
        size
        node {{
          name
        }}
      }}
    }}
    
    licenseInfo {{
      name
      spdxId
    }}
    
    owner {{
      login
      __typename
    }}
    
    defaultBranchRef {{
      name
      target {{
        ... on Commit {{
          history(first: 30) {{
            nodes {{
              committedDate
            }}
          }}
        }}
      }}
    }}
    
    readme1: object(expression: "HEAD:README.md") {{
      ... on Blob {{
        text
      }}
    }}
    readme2: object(expression: "HEAD:readme.md") {{
      ... on Blob {{
        text
      }}
    }}
    readme3: object(expression: "HEAD:README.rst") {{
      ... on Blob {{
        text
      }}
    }}
    readme4: object(expression: "HEAD:README.txt") {{
      ... on Blob {{
        text
      }}
    }}
    readme5: object(expression: "HEAD:README") {{
      ... on Blob {{
        text
      }}
    }}

    stargazers(last: 100) {{
      edges {{
        starredAt
      }}
    }}
    
    mentionableUsers(first: 100) {{
      totalCount
      nodes {{
        login
      }}
    }}
  }}"""
        query_parts.append(part)
    
    query_parts.append("""
  rateLimit {
    cost
    remaining
    resetAt
  }
}""")
    
    return "\n".join(query_parts)
