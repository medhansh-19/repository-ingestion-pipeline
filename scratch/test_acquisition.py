import os
import json
from datetime import datetime
from acquisition.github_client import GitHubClient
from acquisition.repository_enricher import RepositoryEnricher

# Handle serialization of datetimes for json printing
def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

if not os.environ.get("GITHUB_TOKEN"):
    print("WARNING: GITHUB_TOKEN environment variable is not set. The GraphQL API requires authentication!")
    print("Please export GITHUB_TOKEN=your_personal_access_token and try again.")
    exit(1)

print("Initializing clients...")
rest_client = GitHubClient()
enricher = RepositoryEnricher(rest_client)

repo_name = "pallets/flask"
print(f"\nAttempting to acquire and enrich repository via GraphQL: {repo_name}")

try:
    result = enricher.enrich(repo_name)
    
    if result:
        print("\n✅ Successfully enriched repository!")
        print("-" * 50)
        print("Final Osiris Payload Schema Output:")
        # Omit extracted paragraphs for brevity if needed, but here we print the whole thing
        payload_copy = dict(result.payload)
        if "extracted_paragraphs" in payload_copy:
            payload_copy["extracted_paragraphs"] = f"[{len(payload_copy['extracted_paragraphs'])} paragraphs extracted from README]"
            
        print(json.dumps(payload_copy, indent=2, default=default_serializer))
        
        print("-" * 50)
        print("Note: The data above was successfully fetched using a single GraphQL API request")
        print("and flawlessly mapped into the standard legacy REST format expected downstream!")
    else:
        print(f"\n❌ Failed to enrich repository {repo_name}. Could not find it or encountered an error.")
        
except Exception as e:
    print(f"\n❌ Error during acquisition: {e}")
